import sys
import h5py

class H5Writer(object):
    def __init__(self, resize_in_events=1000, number_of_consecutive_events_per_rank=10, debug=False):
        self.initialLength = resize_in_events
        self.number_of_consecutive_events_per_rank = number_of_consecutive_events_per_rank
        self.datasetsThatGrow = []
        self.nextEntryToGrow = resize_in_events
        self.debug=debug
        
    def extendDatasetsThatGrowIfNeeded(self, entryNumber):
        assert entryNumber <= self.nextEntryToGrow, "entryNumber=%d is greater than nextEntryToGrow, entryNumber must be consecutive in calls to this function" % entryNumber
        if entryNumber < self.nextEntryToGrow: return
        assert entryNumber == self.nextEntryToGrow
        for dsetDict in self.datasetsThatGrow:
            dset, current_size = dsetDict['dset'], dsetDict['size']
            assert current_size[0]==entryNumber, "The dset=%s has a current size=%s, first dimensions is not %d" % (dset.name, current_size, entryNumber)
            new_size = list(current_size)
            new_size[0] += self.initialLength
            ## collective call:
            dset.resize(new_size)
            dsetDict['size']=tuple(new_size)
            self.dprint("resized dset=%s to %s" % (dset.name, new_size))
        self.nextEntryToGrow += self.initialLength

    def createFile(self, fname, comm):
        self.fname = fname
        self.comm=comm
        if self.comm.size==1:
            self.h5 = h5py.File(fname, 'w')
        else:
            self.h5 = h5py.File(fname, 'w', driver='mpio', comm=comm)
        self.dprint('created file: %s' % fname)

    def dprint(self, msg):
        if self.debug and self.comm.rank==0:
            print msg
            sys.stdout.flush()

    def thisRankWritesThisEvent(self, entryNumber):
        blockNumber = entryNumber // self.number_of_consecutive_events_per_rank
        assert isinstance(blockNumber, int)
        return blockNumber % self.comm.size == self.comm.rank
    
    def createDatasetThatGrows(self, h5path, shape, chunks, dtype):        
        assert shape[0]==None, "start shape with None to indicate that it grows in that dimension"
        assert h5path.startswith('/'), "h5path must start at root"
        flds=h5path.split('/')
        assert flds[0]==''
        flds.pop(0)
        assert len(flds)>=1, "no h5path given"
        datasetName = flds.pop()
        
        self.dprint("createDatasetThatGrows h5path=%s shape=%s chunks=%s" % (h5path, shape, chunks))
        gr = self.h5['/']
        self.dprint("  %s" % gr.name)
        for fld in flds:
            self.dprint("  fld=%s" % fld)
            if fld not in gr.keys():
                self.dprint("attempting to create group %s" % fld)
                gr.create_group(fld)
            gr = gr[fld]
        initial_shape = [self.initialLength] + list(shape[1:])
        self.dprint("attempting to create dataset %s" % datasetName)
        dset =  gr.create_dataset(datasetName, tuple(initial_shape), dtype=dtype, chunks=chunks, maxshape=shape) 
        self.datasetsThatGrow.append({'dset':dset, 'size':tuple(initial_shape)})
        return dset
    
    def trimDatasetsThatGrowTo(self, finalLen):
        for dsetDict in self.datasetsThatGrow:
            dset, current_size = dsetDict['dset'], dsetDict['size']
            final_size= list(current_size)
            final_size[0]=finalLen
            dset.resize(final_size)
            self.dprint('resized dataset %s to %s' % (dset.name, final_size))

    def close(self):
        self.h5.close()

        
