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



def load_example(name='ab701a5a-2dc3-11eb-9890-0242ac110002'):
    # load data (including the barcode table B)
    import pkg_resources
    import h5py
    from . import benchmarks
    DATA_PATH = pkg_resources.resource_filename('bardensr.benchmarks', f'{name}.hdf5')

    return benchmarks.load_h5py(DATA_PATH)
