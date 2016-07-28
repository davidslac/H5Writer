import os
import sys
import time
import argparse
import traceback

import psana
import numpy as np
from mpi4py import MPI
from H5Writer import H5Writer

comm = MPI.COMM_WORLD
DEBUG = False

############# arguments

parser = argparse.ArgumentParser(description='Process arguments to use_H5Writer.')

parser.add_argument('--run', help='integer of run number to convert', type=int)
parser.add_argument('--h5resize', help='resize value for h5 entries, defaults to 1000', default=1000, type=int)
parser.add_argument('--h5consecutive', help='number of events for each rank to handle consecutively. Defaults to 10', default=10, type=int)
parser.add_argument('--expname', help='experiment name, i.e, xppm0416', type=str)
parser.add_argument('--ffb', help='true if have fast feedback (you are at the beam), defaults to False', default=False, action='store_true')
parser.add_argument('--force', help='overrwite existing file', default=False, action='store_true')
parser.add_argument('--debug', help='debug output', action='store_true', default=False)
parser.add_argument('--lfs_stripe_count', help='set a lfs stripe count on the output file. Defaults to 16', default=16, type=int)
parser.add_argument('--nevents', help='for testing, specify a small number of events to process', default=0, type=int)
defOutDir = "/reg/d/psdm/xpp/xpp06616/scratch/test_h5writer"
parser.add_argument('--outdir', help='output prefix. Include trailing / for directory. expname-run.h5 will be added. Defaults to %s' % defOutDir, 
                    type=str, default=defOutDir)

args = parser.parse_args()
assert args.run, "must provide a run, the --run option"
assert args.expname, "must provide a experiment name for input, the --expanme option"
DEBUG = args.debug

if comm.rank==0:
    msg = 'comm.size=%d program.args: ' % comm.size
    for param in dir(args):
        if param.startswith('_'): continue
        msg += ' %s=%s' % (param, getattr(args, param))
    print msg
    sys.stdout.flush()
    
############# functions
def dprint(msg, tm=True):
    if not DEBUG: return
    hdr = "rank=%4d: " % comm.rank
    if tm:
        hdr += "tm=%10.5f " % time.time()
    print "%s %s" % (hdr, msg)

def eventHasCsPad(evt):
    for ky in evt.keys():
        if ky.type() is psana.CsPad.DataV2:
            return True
    return False

def doNotWrite(evt):
    res = eventHasCsPad(evt)
#    dprint("evt: %s hasCspad=%s" % (evt.get(psana.EventId), res), tm=False) 
    return not res

############# init datasource and h5writer
datasourceString = 'exp=%s:run=%d:smd:live' % (args.expname, args.run)
if args.ffb:
    instr = args.expname[0:3]
    datasourceString += ':dir=/reg/d/ffb/%s/%s/xtc' % (instr, args.expname)
ds = psana.DataSource(datasourceString)
if comm.rank==0: print "datasource: %s" % datasourceString 

h5out = H5Writer(resize_in_events=args.h5resize, 
                 number_of_consecutive_events_per_rank=args.h5consecutive,
                 debug=DEBUG)
outputfile = os.path.join(args.outdir, "%s_r%4.4d.h5" % (args.expname, args.run))
if comm.rank==0:
    if os.path.exists(outputfile):
        if args.force:
            print "removing old output file %s" % outputfile
            os.unlink(outputfile)            
comm.Barrier()
time.sleep(1)
if os.path.exists(outputfile):
    print "output file %s exists, use --force to overwrite" % outputfile
    sys.exit(1)
    
if comm.size > 0 and args.lfs_stripe_count > 0:
    if comm.rank == 0:
        cmd = "lfs setstripe --count %d %s" % (args.lfs_stripe_count, outputfile)
        os.system(cmd)
        print "ran %s" % cmd
    comm.Barrier()
time.sleep(1)
    
h5out.createFile(fname=outputfile, comm=comm)
if comm.rank==0: print "output file: %s" % outputfile
    
############### event loop
entryNumber = -1

cspad_det = psana.Detector('cspad')
ebeam_det = psana.Detector('EBeam')
cspad_shape = None
cspad_h5_dset = None
ebeam_h5_dset = None

bigdata_chunksize=30  # with 850*850*4 bytes per cspad, chunksize=30 is 82MB chunks
smalldata_chunksize=1024

t0 = time.time()
try:
    for evtNum, evt in enumerate(ds.events()):
        if (args.nevents > 0) and (evtNum >= args.nevents): 
            break
        if comm.rank==0 and evtNum % 1000==0 and evtNum > 1:
            print "rank 0: evtNum=%6d entryNumber=%6d rate=%.2f hz" % (evtNum, entryNumber, evtNum/float(time.time()-t0))
        if doNotWrite(evt): continue
        entryNumber += 1

        if cspad_shape is None:
            img = cspad_det.image(evt)
            assert img is not None
            img = img[0:850,0:850]
            cspad_shape = img.shape
            cspad_h5_dset = h5out.createDatasetThatGrows(h5path='/entry_1/instrument_1/detector_1/data',
                                                         shape=(None, cspad_shape[0], cspad_shape[1]),
                                                         chunks=(bigdata_chunksize, cspad_shape[0], cspad_shape[1]),
                                                         dtype=np.float32)
            ebeam_h5_dset = h5out.createDatasetThatGrows(h5path='/entry_1/instrument_1/source_1/energy_eV',
                                                         shape=(None,),
                                                         chunks=(smalldata_chunksize,),
                                                         dtype=np.float32)

            eventnum_dset = h5out.createDatasetThatGrows(h5path='/entry_1/instrument_1/detector_1/eventnum',
                                                         shape=(None,),
                                                         chunks=(smalldata_chunksize,),
                                                         dtype=np.float32)

        h5out.extendDatasetsThatGrowIfNeeded(entryNumber)
        if not h5out.thisRankWritesThisEvent(entryNumber): continue
        img = cspad_det.image(evt)
        assert img is not None
        img = img[0:850,0:850]
        ebeam = ebeam_det.get(evt)

        if ebeam is None:
            energy = 0.0
        else:
            energy = ebeam.ebeamL3Energy()
        dprint("writing to entryNumber=%d" % entryNumber)
        cspad_h5_dset[entryNumber]=img
        ebeam_h5_dset[entryNumber]=energy
        eventnum_dset[entryNumber]=evtNum

    h5out.trimDatasetsThatGrowTo(entryNumber+1)
    h5out.close()

    comm.Barrier()
    if comm.rank==0:
        total_time = time.time()-t0
        hertz = entryNumber / float(total_time)
        assert len(cspad_shape)==2
        total_bytes_written = 4 * entryNumber * cspad_shape[0] * cspad_shape[1]
        total_mbytes = total_bytes_written / float(1024*1024)
        mb_sec = total_mbytes / float(total_time)
        print "-------- finished ---------"
        print "-- mpi.size=%d" % comm.size
        print "-- outputfile=%s" % outputfile
        print "-- total_time = %.2f minutes. Wrote %d events." % (total_time/60.0, evtNum)
        print "-- %.2f hertz  %.1f MB/sec (%.3f GB total)" % (hertz, mb_sec, total_mbytes/float(1024))

        lscmd = 'ls -lrth %s' % outputfile
        print lscmd
        sys.stdout.flush()
        os.system(lscmd)
        sys.stdout.flush()

        getstripecmd = 'lfs getstripe %s' % outputfile
        print getstripecmd
        sys.stdout.flush()
        os.system(getstripecmd)
        sys.stdout.flush()

        os.unlink(outputfile)

except Exception:
    traceback.print_exc()
    comm.Abort(1)
