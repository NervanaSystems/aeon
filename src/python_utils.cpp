#include "python_utils.hpp"

namespace nervana
{
    namespace python
    {
#ifdef PYTHON_PLUGIN
        bool IsLockedGIL()
        {
#if PY_MAJOR_VERSION >= 3
            return PyGILState_Check();
#else
            PyThreadState* tstate = _PyThreadState_Current;
            return tstate && (tstate == PyGILState_GetThisThreadState());
#endif
        }

        static_initialization::static_initialization()
        {
            if (!IsLockedGIL())
            {
                Py_Initialize();
                PyEval_InitThreads();
                _save = PyEval_SaveThread();
            }
        }

        static_initialization::~static_initialization()
        {
            if (_save)
                PyEval_RestoreThread(_save);
        }
        namespace
        {
            static auto& instance = python::static_initialization::Instance();
        }

        allow_threads::allow_threads()
            : _state{PyEval_SaveThread()}
        {
        }

        allow_threads::~allow_threads() { PyEval_RestoreThread(_state); }
        block_threads::block_threads(allow_threads& a)
            : _parent{a}
        {
            std::swap(_state, _parent._state);
            PyEval_RestoreThread(_state);
        }

        block_threads::~block_threads()
        {
            PyEval_SaveThread();
            std::swap(_parent._state, _state);
        }
#endif
    }
}
