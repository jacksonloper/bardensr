__all__=['example_simulation']

from . import programs
from . import kernels
from . import diagnostics
from . import singlefov
from . import plotting
from . import rectangles
from . import memcheck
from . import misc

def ipydoc(x,saynm=True,strip4=True):
    import IPython.display
    nm=x.__name__
    docstr=x.__doc__

    if strip4:
        docstr='\n'.join([x[4:] for x in docstr.split('\n')])

    if saynm:
        return IPython.display.Markdown(f'''\n ### `'''+nm + '`\n' + docstr)
    else:
        return IPython.display.Markdown(docstr)
