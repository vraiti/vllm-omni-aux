/*
 * call_trace: a trace function for sys.settrace that logs every call of a
 * named function of one module, per process, thread, and coroutine.
 *
 *   import sys, call_trace
 *   sys.settrace(call_trace.call_trace())
 *
 * call_trace(module=None, directory="/tmp/logs/call-trace") returns the trace
 * function; module defaults to the top-level package of the caller (so
 * calling it from vllm_omni traces vllm_omni and its submodules).
 *
 * It only acts on "call" events and returns None, so traced frames produce no
 * further line/return events. Every call to a function of that module whose
 * qualified name is not anonymous (contains no '<' or '>': <lambda>,
 * <genexpr>, <module>, nested <locals> functions) appends
 *
 *   (<pid>, <thread>, <coro>) <module>.<qualname>
 *
 * to <directory>/call-trace_<pid>_<thread>_<coro>.log, created (overwriting
 * any old file) the first time it is needed:
 *
 *   thread  the OS thread id (gettid);
 *   coro    the outermost coroutine on the current stack (for asyncio, the
 *           task's coroutine), numbered from 1 in order of first use per
 *           process; 0 is the "null coroutine", plain synchronous execution
 *           with no coroutine above it. Each file starts with a '#' header
 *           naming its root coroutine. A coroutine's files are closed when it
 *           is garbage-collected.
 *
 * A generator/coroutine frame is logged once, when it first starts: its
 * resumptions and throws (including the close() of its finalizer) also
 * arrive as "call" events, and are told apart by the frame's position.
 *
 * sys.settrace covers the calling thread only; use
 * threading.settrace_all_threads(tracer) (3.12+) for every thread. Each process
 * has to install its own tracer; a tracer inherited through fork() starts over
 * with the child's pid.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>
#include <stdio.h>
#include <unistd.h>

#define FILE_CAPSULE "call_trace.file"

typedef struct {
    PyObject_HEAD
    PyObject *prefix;       /* str: traced module */
    PyObject *prefix_dot;   /* str: traced module + "." */
    PyObject *code_cache;   /* dict: code -> fully-qualified name | None */
    PyObject *start_cache;  /* dict: generator code -> byte offset of its first RESUME */
    PyObject *gen_ids;      /* dict: id(coroutine) -> coroutine number */
    PyObject *gen_weakrefs; /* dict: id(gen) -> weakref whose callback forgets it */
    PyObject *files;        /* dict: coroutine number -> {thread id -> FILE capsule} */
    long long next_coro;
    long pid;
    char dir[4096];
} Tracer;

static PyTypeObject TracerType;

/* ------------------------------------------------------------------------ */
/* Which functions are logged                                                */
/* ------------------------------------------------------------------------ */

static int
module_matches(Tracer *self, PyObject *name)
{
    return PyUnicode_Compare(name, self->prefix) == 0
           || PyUnicode_Tailmatch(name, self->prefix_dot, 0, PY_SSIZE_T_MAX, -1) == 1;
}

static int
is_anonymous(PyObject *qualname)
{
    Py_ssize_t len = PyUnicode_GET_LENGTH(qualname);
    return PyUnicode_FindChar(qualname, '<', 0, len, 1) >= 0 || PyUnicode_FindChar(qualname, '>', 0, len, 1) >= 0;
}

/* New reference: the frame's "<module>.<qualname>" if it is logged, else None. */
static PyObject *
qualified_name(Tracer *self, PyFrameObject *frame)
{
    PyCodeObject *code = PyFrame_GetCode(frame);
    PyObject *cached;
    int found = PyDict_GetItemRef(self->code_cache, (PyObject *)code, &cached);
    if (found != 0) {
        Py_DECREF(code);
        return found < 0 ? NULL : cached;
    }

    PyObject *result = Py_NewRef(Py_None);
    PyObject *globals = PyFrame_GetGlobals(frame);
    PyObject *module = NULL;
    PyObject *qualname = PyObject_GetAttrString((PyObject *)code, "co_qualname");
    if (qualname == NULL) {
        PyErr_Clear();
    }
    if (globals != NULL && PyDict_Check(globals) && PyDict_GetItemStringRef(globals, "__name__", &module) < 0) {
        PyErr_Clear();
    }
    /* CO_OPTIMIZED: a function body, not a class body or module. */
    if ((code->co_flags & CO_OPTIMIZED) && module != NULL && PyUnicode_Check(module) && qualname != NULL
        && PyUnicode_Check(qualname) && module_matches(self, module) && !is_anonymous(qualname)) {
        PyObject *name = PyUnicode_FromFormat("%U.%U", module, qualname);
        if (name != NULL) {
            Py_SETREF(result, name);
        }
    }
    Py_XDECREF(globals);
    Py_XDECREF(module);
    Py_XDECREF(qualname);
    if (PyErr_Occurred() || PyDict_SetItem(self->code_cache, (PyObject *)code, result) < 0) {
        Py_DECREF(code);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(code);
    return result;
}

/* ------------------------------------------------------------------------ */
/* Generators and coroutines                                                 */
/* ------------------------------------------------------------------------ */

/* Weakref callback, self = (tracer, id(coroutine)): forget a dead root
   coroutine and close its files, so descriptors do not accumulate. */
static PyObject *
gen_died(PyObject *bound, PyObject *Py_UNUSED(weakref))
{
    Tracer *self = (Tracer *)PyTuple_GET_ITEM(bound, 0);
    PyObject *key = PyTuple_GET_ITEM(bound, 1);
    if (self->gen_ids != NULL) {
        PyObject *number = NULL;
        if (PyDict_Pop(self->gen_ids, key, &number) == 1) {
            if (PyLong_AsLongLong(number) > 0 && self->files != NULL) {
                PyDict_Pop(self->files, number, NULL);
            }
            Py_DECREF(number);
        }
    }
    if (self->gen_weakrefs != NULL) {
        PyDict_Pop(self->gen_weakrefs, key, NULL);
    }
    PyErr_Clear();
    Py_RETURN_NONE;
}

static PyMethodDef gen_died_def = {"_gen_died", (PyCFunction)gen_died, METH_O, NULL};

/* The number of a root coroutine, assigned on first sight. Returns 0, or
   -1 on error. */
static int
coro_number(Tracer *self, PyObject *coro, long long *number)
{
    PyObject *key = PyLong_FromVoidPtr(coro);
    if (key == NULL) {
        return -1;
    }
    PyObject *value;
    int found = PyDict_GetItemRef(self->gen_ids, key, &value);
    if (found != 0) {
        if (found > 0) {
            *number = PyLong_AsLongLong(value);
            Py_DECREF(value);
        }
        Py_DECREF(key);
        return found < 0 ? -1 : 0;
    }

    *number = self->next_coro++;
    PyObject *bound = PyTuple_Pack(2, (PyObject *)self, key);
    PyObject *callback = bound ? PyCFunction_New(&gen_died_def, bound) : NULL;
    PyObject *weakref = callback ? PyWeakref_NewRef(coro, callback) : NULL;
    PyObject *id = PyLong_FromLongLong(*number);
    int ok = weakref != NULL && id != NULL && PyDict_SetItem(self->gen_weakrefs, key, weakref) == 0
             && PyDict_SetItem(self->gen_ids, key, id) == 0;
    Py_XDECREF(bound);
    Py_XDECREF(callback);
    Py_XDECREF(weakref);
    Py_XDECREF(id);
    Py_DECREF(key);
    return ok ? 0 : -1;
}

static int g_resume_opcode = -1; /* dis.opmap["RESUME"] */

/* Byte offset of a generator code's first RESUME, where its frame is when it
   first starts (it follows RETURN_GENERATOR and POP_TOP); every resumption or
   throw is at a later RESUME. Cached per code object; -1 on error. */
static int
start_offset(Tracer *self, PyCodeObject *code)
{
    PyObject *cached;
    int found = PyDict_GetItemRef(self->start_cache, (PyObject *)code, &cached);
    if (found != 0) {
        int offset = found < 0 ? -1 : (int)PyLong_AsLong(cached);
        Py_XDECREF(cached);
        return offset;
    }
    PyObject *bytecode = PyCode_GetCode(code);
    if (bytecode == NULL) {
        return -1;
    }
    const unsigned char *raw = (const unsigned char *)PyBytes_AS_STRING(bytecode);
    Py_ssize_t size = PyBytes_GET_SIZE(bytecode);
    int offset = -2; /* no RESUME: never matches a real position */
    for (Py_ssize_t i = 0; i + 1 < size; i += 2) {
        if (raw[i] == g_resume_opcode) {
            offset = (int)i;
            break;
        }
    }
    Py_DECREF(bytecode);
    PyObject *value = PyLong_FromLong(offset);
    if (value == NULL || PyDict_SetItem(self->start_cache, (PyObject *)code, value) < 0) {
        Py_XDECREF(value);
        return -1;
    }
    Py_DECREF(value);
    return offset;
}

/* New reference to the outermost coroutine on the frame's stack, or NULL. */
static PyObject *
root_coroutine(PyFrameObject *frame)
{
    PyObject *root = NULL;
    PyFrameObject *f = (PyFrameObject *)Py_NewRef(frame);
    while (f != NULL) {
        PyObject *gen = PyFrame_GetGenerator(f);
        if (gen != NULL && PyCoro_CheckExact(gen)) {
            Py_XSETREF(root, gen);
        }
        else {
            Py_XDECREF(gen);
        }
        PyFrameObject *back = PyFrame_GetBack(f);
        Py_DECREF(f);
        f = back;
    }
    return root;
}

/* ------------------------------------------------------------------------ */
/* Log files                                                                 */
/* ------------------------------------------------------------------------ */

static void
close_file(PyObject *capsule)
{
    FILE *fp = PyCapsule_GetPointer(capsule, FILE_CAPSULE);
    if (fp != NULL) {
        fclose(fp);
    }
}

/* After fork() the child must not write into the parent's files or reuse its
   coroutine numbers: start over under the child's pid. */
static void
check_pid(Tracer *self)
{
    long pid = (long)getpid();
    if (pid == self->pid) {
        return;
    }
    self->pid = pid;
    self->next_coro = 1;
    PyDict_Clear(self->files);
    PyDict_Clear(self->gen_ids);
    PyDict_Clear(self->gen_weakrefs);
}

/* New reference to the FILE capsule of (thread, coro), creating it if needed. */
static PyObject *
log_file(Tracer *self, unsigned long thread, long long coro, PyObject *root)
{
    PyObject *coro_key = PyLong_FromLongLong(coro);
    PyObject *thread_key = PyLong_FromUnsignedLong(thread);
    PyObject *per_thread = NULL;
    PyObject *capsule = NULL;
    if (coro_key == NULL || thread_key == NULL) {
        goto done;
    }
    int found = PyDict_GetItemRef(self->files, coro_key, &per_thread);
    if (found < 0) {
        goto done;
    }
    if (found == 0) {
        per_thread = PyDict_New();
        if (per_thread == NULL || PyDict_SetItem(self->files, coro_key, per_thread) < 0) {
            goto done;
        }
    }
    if (PyDict_GetItemRef(per_thread, thread_key, &capsule) != 0) {
        goto done;
    }

    char path[4200];
    snprintf(path, sizeof(path), "%s/call-trace_%ld_%lu_%lld.log", self->dir, self->pid, thread, coro);
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        goto done;
    }
    /* Line-buffered: a killed process (servers usually end that way) keeps
       everything it logged. */
    setvbuf(fp, NULL, _IOLBF, 0);
    if (root != NULL) {
        PyObject *name = PyObject_GetAttrString(root, "__qualname__");
        const char *text = name != NULL && PyUnicode_Check(name) ? PyUnicode_AsUTF8(name) : NULL;
        fprintf(fp, "# pid=%ld thread=%lu coro=%lld root=%s\n", self->pid, thread, coro, text ? text : "?");
        Py_XDECREF(name);
        PyErr_Clear();
    }
    else {
        fprintf(fp, "# pid=%ld thread=%lu coro=0 (no coroutine)\n", self->pid, thread);
    }
    capsule = PyCapsule_New(fp, FILE_CAPSULE, close_file);
    if (capsule == NULL) {
        fclose(fp);
        goto done;
    }
    if (PyDict_SetItem(per_thread, thread_key, capsule) < 0) {
        Py_CLEAR(capsule);
    }

done:
    Py_XDECREF(coro_key);
    Py_XDECREF(thread_key);
    Py_XDECREF(per_thread);
    return capsule;
}

/* ------------------------------------------------------------------------ */
/* The trace function                                                        */
/* ------------------------------------------------------------------------ */

static void
record_call(Tracer *self, PyFrameObject *frame)
{
    PyObject *name = qualified_name(self, frame);
    if (name == NULL || name == Py_None) {
        Py_XDECREF(name);
        return;
    }
    check_pid(self);

    PyCodeObject *code = PyFrame_GetCode(frame);
    int is_generator = code->co_flags & (CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR);
    int start = is_generator ? start_offset(self, code) : 0;
    Py_DECREF(code);
    if (is_generator && (start == -1 || PyFrame_GetLasti(frame) != start)) {
        /* A resumption or throw, not the first start (or an error). */
        Py_DECREF(name);
        return;
    }

    long long coro = 0;
    PyObject *root = root_coroutine(frame);
    if (root != NULL && coro_number(self, root, &coro) < 0) {
        Py_DECREF(root);
        Py_DECREF(name);
        return;
    }
    unsigned long thread = PyThread_get_thread_native_id();
    PyObject *capsule = log_file(self, thread, coro, root);
    if (capsule != NULL) {
        FILE *fp = PyCapsule_GetPointer(capsule, FILE_CAPSULE);
        const char *text = PyUnicode_AsUTF8(name);
        if (fp != NULL && text != NULL) {
            fprintf(fp, "(%ld, %lu, %lld) %s\n", self->pid, thread, coro, text);
        }
        Py_DECREF(capsule);
    }
    Py_XDECREF(root);
    Py_DECREF(name);
}

/* tracer(frame, event, arg): the sys.settrace protocol. Always returns None,
   so no frame gets a local trace function (no line/return events). */
static PyObject *
tracer_call(Tracer *self, PyObject *args, PyObject *kwargs)
{
    PyObject *frame, *event, *arg;
    if (!PyArg_ParseTuple(args, "OOO", &frame, &event, &arg)) {
        return NULL;
    }
    if (PyFrame_Check(frame) && PyUnicode_Check(event) && PyUnicode_CompareWithASCIIString(event, "call") == 0) {
        /* Never disturb the traced program: report our own failures as
           unraisable rather than raising them into the traced code (which
           would also make sys.settrace drop this tracer). */
        PyObject *pending = PyErr_GetRaisedException();
        record_call(self, (PyFrameObject *)frame);
        if (PyErr_Occurred()) {
            PyErr_WriteUnraisable((PyObject *)self);
        }
        PyErr_SetRaisedException(pending);
    }
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------------ */
/* Type and module                                                           */
/* ------------------------------------------------------------------------ */

static int
tracer_traverse(Tracer *self, visitproc visit, void *arg)
{
    Py_VISIT(self->prefix);
    Py_VISIT(self->prefix_dot);
    Py_VISIT(self->code_cache);
    Py_VISIT(self->start_cache);
    Py_VISIT(self->gen_ids);
    Py_VISIT(self->gen_weakrefs);
    Py_VISIT(self->files);
    return 0;
}

static int
tracer_clear(Tracer *self)
{
    Py_CLEAR(self->prefix);
    Py_CLEAR(self->prefix_dot);
    Py_CLEAR(self->code_cache);
    Py_CLEAR(self->start_cache);
    Py_CLEAR(self->gen_ids);
    Py_CLEAR(self->gen_weakrefs);
    Py_CLEAR(self->files);
    return 0;
}

static void
tracer_dealloc(Tracer *self)
{
    PyObject_GC_UnTrack(self);
    tracer_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
tracer_repr(Tracer *self)
{
    return PyUnicode_FromFormat("<call_trace module=%R directory=%s>", self->prefix, self->dir);
}

static PyTypeObject TracerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "call_trace.CallTracer",
    .tp_basicsize = sizeof(Tracer),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_doc = "Trace function for sys.settrace; create with call_trace.call_trace().",
    .tp_call = (ternaryfunc)tracer_call,
    .tp_repr = (reprfunc)tracer_repr,
    .tp_traverse = (traverseproc)tracer_traverse,
    .tp_clear = (inquiry)tracer_clear,
    .tp_dealloc = (destructor)tracer_dealloc,
};

/* The top-level package of the module that called call_trace(). */
static PyObject *
caller_package(void)
{
    PyObject *globals = PyEval_GetFrameGlobals();
    PyObject *name = NULL;
    if (globals == NULL || PyDict_GetItemStringRef(globals, "__name__", &name) <= 0 || !PyUnicode_Check(name)) {
        Py_XDECREF(globals);
        Py_XDECREF(name);
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "call_trace(): cannot determine the calling module");
        }
        return NULL;
    }
    Py_DECREF(globals);
    Py_ssize_t dot = PyUnicode_FindChar(name, '.', 0, PyUnicode_GET_LENGTH(name), 1);
    PyObject *package = dot < 0 ? Py_NewRef(name) : (dot == -2 ? NULL : PyUnicode_Substring(name, 0, dot));
    Py_DECREF(name);
    return package;
}

static int
make_directory(const char *directory)
{
    PyObject *os = PyImport_ImportModule("os");
    PyObject *makedirs = os ? PyObject_GetAttrString(os, "makedirs") : NULL;
    PyObject *args = Py_BuildValue("(s)", directory);
    PyObject *kwargs = Py_BuildValue("{s:O}", "exist_ok", Py_True);
    PyObject *made = makedirs && args && kwargs ? PyObject_Call(makedirs, args, kwargs) : NULL;
    Py_XDECREF(os);
    Py_XDECREF(makedirs);
    Py_XDECREF(args);
    Py_XDECREF(kwargs);
    Py_XDECREF(made);
    return made == NULL ? -1 : 0;
}

static PyObject *
call_trace(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwargs)
{
    static char *keywords[] = {"module", "directory", NULL};
    PyObject *module = Py_None;
    const char *directory = "/tmp/logs/call-trace";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Os", keywords, &module, &directory)) {
        return NULL;
    }
    if (strlen(directory) >= sizeof(((Tracer *)0)->dir)) {
        PyErr_SetString(PyExc_ValueError, "directory path is too long");
        return NULL;
    }
    PyObject *prefix = module == Py_None ? caller_package() : Py_NewRef(module);
    if (prefix == NULL) {
        return NULL;
    }
    if (!PyUnicode_Check(prefix)) {
        Py_DECREF(prefix);
        PyErr_SetString(PyExc_TypeError, "module must be a str");
        return NULL;
    }
    if (make_directory(directory) < 0) {
        Py_DECREF(prefix);
        return NULL;
    }
    Tracer *self = PyObject_GC_New(Tracer, &TracerType);
    if (self == NULL) {
        Py_DECREF(prefix);
        return NULL;
    }
    self->prefix = prefix;
    self->prefix_dot = PyUnicode_FromFormat("%U.", prefix);
    self->code_cache = PyDict_New();
    self->start_cache = PyDict_New();
    self->gen_ids = PyDict_New();
    self->gen_weakrefs = PyDict_New();
    self->files = PyDict_New();
    self->next_coro = 1;
    self->pid = (long)getpid();
    strcpy(self->dir, directory);
    PyObject_GC_Track(self);
    if (self->prefix_dot == NULL || self->code_cache == NULL || self->start_cache == NULL || self->gen_ids == NULL
        || self->gen_weakrefs == NULL || self->files == NULL) {
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject *)self;
}

static PyMethodDef methods[] = {
    {"call_trace", (PyCFunction)(void (*)(void))call_trace, METH_VARARGS | METH_KEYWORDS,
     "call_trace(module=None, directory='/tmp/logs/call-trace') -> trace function\n\n"
     "Install with sys.settrace(call_trace.call_trace()); module defaults to the caller's top-level package."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "call_trace",
    "Per-(pid, thread, coroutine) call logs for one module's named functions.",
    -1,
    methods,
};

PyMODINIT_FUNC
PyInit_call_trace(void)
{
    if (PyType_Ready(&TracerType) < 0) {
        return NULL;
    }
    PyObject *opcode = PyImport_ImportModule("opcode");
    PyObject *opmap = opcode ? PyObject_GetAttrString(opcode, "opmap") : NULL;
    PyObject *resume = opmap ? PyMapping_GetItemString(opmap, "RESUME") : NULL;
    Py_XDECREF(opcode);
    Py_XDECREF(opmap);
    if (resume == NULL) {
        return NULL;
    }
    g_resume_opcode = (int)PyLong_AsLong(resume);
    Py_DECREF(resume);
    PyObject *module = PyModule_Create(&module_def);
    if (module == NULL) {
        return NULL;
    }
    if (PyModule_AddObjectRef(module, "CallTracer", (PyObject *)&TracerType) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
