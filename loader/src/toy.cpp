#include "toy.hpp"
extern "C" {

/*
backend should be a python object that exposes the method:
def consume(self, np_src):
    '''
    Consumes the row major data in np_src then marks writeable FALSE when done
    (meaning that the data can now be written to in the c-module)
    '''
*/

static char module_docstring[] ="";
static char chromicorn_docstring[] = "";
static PyObject *chromicorn_call(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"chromicorn", chromicorn_call, METH_VARARGS, chromicorn_docstring},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC init_chromicorn(void)
{
    PyObject *m = Py_InitModule3("_chromicorn", module_methods, module_docstring);
    if (m == NULL)
        return;
    /* Load `numpy` functionality. */
    import_array();
}

static PyObject *chromicorn_call(PyObject *self, PyObject *args)
{
    PyObject *backend_obj;
    int m, n;
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "iiO", &m, &n, &backend_obj))
        return NULL;

    int nd = 2;
    npy_intp dims[2] = {m, n};
    PyObject *consumeFunc = PyObject_GetAttrString(backend_obj, "consume");

    if (!PyCallable_Check(consumeFunc))
        return NULL;

    float* data = new float[m * n];

    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            data[j*n + i] = j*n + i;

    PyObject* np_ary = PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT, static_cast<void *>(data));

    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, np_ary);

    PyObject* res = PyObject_CallObject(consumeFunc, pArgs);

    // if (res != NULL) Py_DECREF(res);

    /* Clean up. */
    Py_XDECREF(pArgs);
    delete[] data;
    PyObject *ret = Py_BuildValue("i", 1);
    return ret;
}

}
