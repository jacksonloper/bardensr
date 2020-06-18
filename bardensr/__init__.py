__all__=['example_simulation']

from . import programs
from . import kernels
from . import diagnostics

def ipydoc(x):
    import IPython.display
    nm=x.__name__
    docstr=x.__doc__
    docstr='\n'.join([x[4:] for x in docstr.split('\n')])
    return IPython.display.Markdown('''-------\n ### `'''+nm + '`\n' + docstr)
