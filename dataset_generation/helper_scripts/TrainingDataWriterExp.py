import glob
import json
import os
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
import queue
import sys
import numpy as np
import torch
from time import time, sleep
import multiprocessing as mp
import igl
import shutil
from scipy.sparse import save_npz
import threading
import collections
import warnings
from datetime import datetime
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())
import SourceToTargetsFile
from MeshProcessor import MeshProcessor


class TrainingDataWriter:
    def __init__(self, files, target_dir=None, source_dir=None, save_ids=False, file_inds_to_not_process=[], cpus=4,
                 report_times=False, start_file_ind=0, skip_existing=False, single_thread_blas=True,
                 just_copy_missing=False, pairs=None):

        self.all_models = files
        self.cpus = cpus
        self.num_models = len(self.all_models)
        self.target_dir = target_dir
        self.__save_ids = save_ids
        self.__no_processing = file_inds_to_not_process
        self.__report_times = report_times
        self.__start_file_ind = start_file_ind
        self.__skip_existing = skip_existing
        self.__single_thread_blas = single_thread_blas
        self.source_dir = source_dir
        self.__ttype = torch.float
        self.__just_copy_missing = just_copy_missing
        self.__EOF = False
        self.__files_to_dirs = {}
        self.__original_pairs = pairs

    def reading_thread(self, mesh_inds, qmesh: mp.SimpleQueue, startq: queue.SimpleQueue):
        print("READ THREAD LAUNCHED")
        MEMORY = 100
        time_buffer = collections.deque()  # [float('inf')*MEMORY], MEMORY)
        N = len(mesh_inds)
        print_time = time()
        for i, ind in enumerate(mesh_inds):
            if i < self.__start_file_ind:  # skipping files as instructed
                continue
            stime = time()
            startq.put((ind, False, stime))
            self.read_single_mesh(ind, qmesh)
            etime = time() - stime
            if i > qmesh.my_maxsize:  # start timing when queue has filled up
                time_buffer.appendleft(etime)
            if len(time_buffer) > MEMORY:
                time_buffer.pop()  # forget old timings and keep last MEMORY of them

            if time() - print_time > 1:
                print_time = time()
                if len(time_buffer) > 0:
                    avg_time = sum(time_buffer) / len(time_buffer)
                    if avg_time == float('inf'):
                        avg_time = time_buffer[0]
                    finished_for_sure = max(i + 1 - qmesh.my_maxsize, 0)
                    remain_time = (N - finished_for_sure) * avg_time
                    if remain_time < 60:
                        remain_str = f'{remain_time:.0f} seconds'
                    elif remain_time < 3600:
                        remain_str = f'~{remain_time / 60:.1f} minutes'
                    else:
                        remain_str = f'~{remain_time / 3600:.1f} hours'
                    avg_str = f'~{avg_time:.2f} seconds'
                    time_str = f"avg time for each mesh: {avg_str}, tot remaining: {remain_str}"
                    now = datetime.now()
                    curtime = now.strftime("%H:%M:%S")
                    print(f"[{curtime}] Finished at least {finished_for_sure}/{N}: {time_str}")
                else:
                    print(
                        f"**** read thread inserted {i}/{qmesh.my_maxsize} meshes to queue, will start reporting estimated timings once queue is full")
                # if avg_time == float('inf'):
                #
                #     time_str = f"[Ramping up; will start reporting times in {MEMORY - i-1} iterations]"

        print(
            '++++++++ MESH READING THREAD HAS FINISHED READING ALL MESHES, LETTING THE MESH ANALYZER THREAD KNOW THIS IS EOF AND THEN SHUTTING MYSELF')
        qmesh.put(-1)
        self.__EOF = True

    def read_single_mesh(self, mesh_index, qmesh: mp.SimpleQueue):
        stime = time()
        mesh_name = self.all_models[mesh_index]

        if self.__save_ids:
            fname = str(mesh_index).zfill(8)
        else:
            fname = os.path.split(mesh_name)[-1]
            fname = fname.split('.')[0]
        directory = os.path.join(self.target_dir, fname)
        if not self.__just_copy_missing:
            if self.__skip_existing:
                if os.path.exists(directory):
                    file_list = ['vertices.npy', 'faces.npy']
                    if mesh_index not in self.__no_processing:
                        file_list.extend(["samples.npy", "samples_normals.npy", "samples_wks.npy", "centroids_wks.npy",
                                          "centroids_and_normals.npy", 'lap_perm_c.npy', 'lap_perm_r.npy', 'w.npy',
                                          'laplace.npz', 'rhs.npz',
                                          'grad.npz', 'new_grad.npz', 'new_rhs.npz', 'lap_L.npz', 'lap_U.npz'])
                    for sf in file_list:
                        tf = os.path.join(directory, sf)
                        if not os.path.exists(tf):
                            print(f'missing file  {tf} in existing directory, recreating that directory')
                            break
                    else:  # all files accounted for
                        print(
                            f'skipping {directory}, already exists in destination (instructed to skip and not overwrite)')
                        return
                else:
                    print(
                        f'adding {directory}, doesn''t exist in destination (instructed to check if exists before writing)')
            if mesh_name.endswith('off'):
                vertices, faces, _ = igl.read_off(mesh_name)
            elif mesh_name.endswith('obj'):
                vertices, _, _, faces, _, _ = igl.read_obj(mesh_name)
            else:
                raise Exception("unknown file type")
        else:
            vertices = None
            faces = None
        out = {}
        out['mesh_name'] = mesh_name
        out['vertices'] = vertices
        out['faces'] = faces
        out['output_directory'] = directory
        out['mesh_index'] = mesh_index

        if self.__report_times:
            print(f"read time: {time() - stime}")
        stime = time()
        qmesh.put(out)

        if self.__report_times:
            print(f"READ WAITED FOR PROCESSES TO TAKE LOADED MESHES FOR {time() - stime}")

    def mesh_analysis_subprocess(self, read_queue: mp.SimpleQueue, write_queue: mp.SimpleQueue):
        print("SUBPROCESS LAUNCHED")
        if self.__single_thread_blas:
            if os.name != 'nt':  # don't check on windows since in any case this var doesn't work.
                # we want BLAS set to work on only one core
                assert "OPENBLAS_NUM_THREADS" in os.environ and os.environ[
                    "OPENBLAS_NUM_THREADS"] == "1", "if you are on LINUX, you need to export OPENBLAS_NUM_THREADS=1 before running otherwise BLAS takes over all procerssors from one thread "
            else:
                os.environ["OPENBLAS_NUM_THREADS"] = "1"  # this doesn't seem to help but why not try
        # for counter,mesh_ind in enumerate(r):
        while True:
            sstime = time()
            mesh = read_queue.get()

            if self.__report_times:
                print(f"ANALYZER WAITED FOR READER TO GIVE A MESH FOR {time() - sstime}")

            if mesh == -1:  # EOF
                assert read_queue.empty()
                read_queue.put(-1)  # I took the EOF FLAG so need to put it back
                print(
                    '++++++++ MESH ANALYZER SUBPROCESS GOT NOTIFIED THAT NO MORE MESHES ARE TO BE PROCESSED (EOF) SO SHUTTING MYSELF DOWN')
                return
            # s = time()
            vertices = mesh['vertices']
            faces = mesh['faces']
            directory = mesh['output_directory']
            mesh_index = mesh['mesh_index']
            mesh_name = mesh['mesh_name']
            try:
                if self.__report_times:
                    print("ANALYZER GOING TO PROCESS MESH")
                out = self.analyze_single_mesh(vertices, faces, directory, mesh_index, mesh_name)
                if self.__report_times:
                    print("ANALYZER DONE PROCESSING MESH")
                write_queue.put(out)
            except EOFError as e:  # TODO switch to excpetion that is actually catched
                warnings.warn(f'encountered an exception, {e}, skipping mesh for now')
                # print(f"Putting the problematic mesh {mesh_name}, index {mesh_index}, back for processing again")
                # read_queue.put(mesh)

    def analyze_single_mesh(self, vertices, faces, directory, mesh_index, mesh_name):

        out = {}
        out['directory'] = directory
        out['mesh_name'] = mesh_name
        out['mesh_index'] = mesh_index
        should_process = (not self.__just_copy_missing) and (mesh_index not in self.__no_processing)
        processor = MeshProcessor(vertices, faces, self.__ttype, self.source_dir, from_file=True)
        if should_process:
            # processor.compute_everything()
            processor.get_samples()
            processor.get_centroids()
            processor.get_differential_operators()
            # mesh_data = processor.get_source_mesh_data()
        out_np, out_npz = processor.get_writeable()
        out['np'] = out_np
        out['npz'] = out_npz
        sstime = time()

        if self.__report_times:
            print(f"ANALYZER WAITED FOR WRITER TO TAKE COMPUTED DATA FOR {time() - sstime}")
        return out

    def main_writing_thread(self, qout: mp.SimpleQueue, file_copy_queue: queue.SimpleQueue):
        print("WRITE THREAD LAUNCHED")
        while (True):
            stime = time()
            out = qout.get()
            if self.__report_times:
                print(f"WRITER WAITED FOR DATA FROM PROCESSES FOR {time() - stime}")
            stime = time()
            if out == -1:
                assert qout.empty()
                file_copy_queue.put(-1)
                print(
                    '++++++++ WRITING THREAD GOT NOTIFIED THAT NO MORE MESHES ARE WAITING TO BE PROCESSED, LETTING THE COPY THREAD KNOW THIS IS EOF AND THEN SHUTTING MYSELF')
                return
            directory = out['directory']
            mesh_name = out['mesh_name']

            if not os.path.exists(directory):
                os.mkdir(directory)
            if not self.__just_copy_missing:
                np_saves = out['np'].keys()
                for item in np_saves:
                    np.save(os.path.join(directory, item), out['np'][item])

                npz_saves = out['npz'].keys()
                for item in npz_saves:
                    save_npz(os.path.join(directory, item), out['npz'][item])
            file_copy_queue.put((mesh_name, directory, out['mesh_index']))
            if self.__report_times:
                print(f"write time: {time() - stime}")

    def copy_single_dir(self, mesh_name, output_dir):
        mesh_name = mesh_name[0:-4]
        # files = glob.iglob(mesh_name + '_*')
        files = self.relevant[mesh_name]
        for file in files:

            new_name = file.replace(mesh_name + '_', '')
            new_name = os.path.join(output_dir, new_name)
            if self.__just_copy_missing and os.path.exists(new_name):
                continue
            shutil.copy2(file, new_name)
            del file
        del files

    def file_copying_thread(self, file_copy_queue: queue.SimpleQueue, done_queue: queue.SimpleQueue):
        print("COPY THREAD LAUNCHED and processing all files in dir in background")
        # getting all files
        files = os.listdir(self.source_dir)
        files.sort()
        self.relevant = {}
        for file_ind in range(len(files)):
            # check if cur file is obj/off
            if self.__report_times and file_ind % 200 == 0:
                print(f"copy thread: caching dir, currently {file_ind}/{len(files)}")
            fname = files[file_ind]
            if len(fname) < 4 and fname[-4] != '.obj' and fname[-4] != '.off':
                continue
            file = os.path.join(self.source_dir, fname)
            if not os.path.isfile(file):
                continue

            file = os.path.abspath(file)
            # get file name without ending
            key = file[0:-4]
            # find all files that start with "<filename>_"
            additional_files = []
            prefix = key + '_'
            # since we're looking for a prefix and files was sorted we can just go forward, backward
            for next_file_ind in range(file_ind + 1, len(files)):
                next_file = os.path.abspath(os.path.join(self.source_dir, files[next_file_ind]))
                if not next_file.startswith(prefix):
                    break
                additional_files.append(next_file)
            for prev_file_ind in range(file_ind - 1, -1, -1):
                prev_file = os.path.abspath(os.path.join(self.source_dir, files[prev_file_ind]))
                if not prev_file.startswith(prefix):
                    break
                additional_files.append(prev_file)
            self.relevant[key] = additional_files
        # now starting the queue
        print("FILE COPYING THREAD STARTING TO COPY")
        while (True):
            a = file_copy_queue.get()
            if a == -1:
                assert file_copy_queue.empty()
                print('++++++++ FILE COPYING THREAD GOT NOTIFIED NO MORE MESHES TO COPY, SHUTTING MYSELF DOWN')
                return
            mesh_name = a[0]
            full_mesh_name = os.path.abspath(mesh_name)
            output_dir = a[1]
            ind = a[2]
            if self.__report_times:
                print(f'copying {full_mesh_name} to {output_dir}')
            self.copy_single_dir(full_mesh_name, output_dir)
            self.__files_to_dirs[mesh_name] = output_dir
            done_queue.put((ind, True, time()))

    def run(self):
        print(f"CPU {self.cpus}")
        self.block = len(self.all_models) // self.cpus
        threads = []
        r = range(len(self.all_models))
        qin = mp.Queue(maxsize=self.cpus * 2)
        qin.my_maxsize = self.cpus * 2  # hack to get around queue not exposing maxsize
        qout = mp.Queue(maxsize=self.cpus * 2)
        qcopy = mp.SimpleQueue()
        qstartdone = queue.SimpleQueue()
        read_thread = threading.Thread(target=self.reading_thread, args=[r, qin, qstartdone])
        write_thread = threading.Thread(target=self.main_writing_thread, args=[qout, qcopy])
        copy_thread = threading.Thread(target=self.file_copying_thread, args=[qcopy, qstartdone])

        read_thread.start()
        for idx in range(self.cpus):
            o = mp.Process(target=self.mesh_analysis_subprocess, args=[qin, qout])
            o.daemon = True
            threads.append(o)
        for th in threads:
            th.start()
        write_thread.start()
        copy_thread.start()
        print("(MAIN: ALL SUBTHREADS STARTED, GOING TO OVERSEE NONE CRASH")
        # we know that the exit cue was given by read_thread finishing,
        done_meshes = {}
        while (True):
            some_alive = False
            for i, th in enumerate(threads):
                th.join(timeout=1)
                if not th.is_alive():
                    if not self.__EOF:
                        warnings.warn(
                            "XXXXXXXXXXXXXXXXXXXXXXXXXX OH OH, ah process died while the run is not over, gotta start a new one!")
                        o = mp.Process(target=self.mesh_analysis_subprocess, args=[qin, qout])
                        o.daemon = True
                        threads[i] = o
                        threads[i].start()
                        some_alive = True
                else:
                    some_alive = True
                while (not qstartdone.empty()):
                    a = qstartdone.get(timeout=1)
                    ind = a[0]
                    done = a[1]
                    t = a[2]
                    if done:
                        if ind not in done_meshes:
                            warnings.warn(
                                f'XXXXXXXXXXXXXXXX hmmmm, mesh {ind} is reported as done but wasn''t reported as started')
                        else:
                            done_meshes[ind] = True
                    else:  # startinmg
                        if ind in done_meshes:
                            warnings.warn(
                                f'XXXXXXXXXXXXXXXX hmmmm, mesh {ind} is reported as starting but already reported as started')
                        else:
                            done_meshes[ind] = False

            if not some_alive:
                print("All processes ended, looping and waiting for copy thread")
                if qout.empty():  # be a good citizen and also let the writer thread know we're done
                    qout.put(-1)
                sleep(5)
                if copy_thread is None:
                    break
                if not copy_thread.is_alive():  # need to perform one more iteration since this loop has code that processes copy_thread's output, even though it is done
                    copy_thread = None
        last = qin.get()
        assert qin.empty()
        print("MAIN: WAITING FOR WRITE THREAD")
        write_thread.join()
        print('++++++++ MAIN: JOINED ALL SUBPROCESSES/THREADS, EXITED')
        for mesh_ind, done in done_meshes.items():
            if not done:
                warnings.warn(f'FYI!!!!!!!!!!!!!!!!!!!!!! mesh {mesh_ind} has not been reported as done!!!')
        if self.__original_pairs is not None:
            pairs = []
            for pair in self.__original_pairs:
                new_pair = [None, None]
                for i in range(2):
                    if pair[i] not in self.__files_to_dirs:
                        break
                    new_pair[i] = os.path.basename(os.path.normpath(self.__files_to_dirs[pair[i]]))
                if new_pair[0] is None or new_pair[
                    1] is None:  # these did not exist as processed, so there was an issue, and we skip them
                    continue
                pairs.append(new_pair)

            SourceToTargetsFile.write(os.path.join(self.target_dir, 'data.json'), pairs)


def main(sdir: str, tdir: str, skip_target_only=False, cpus=4, verbose=False, skip_to_file_ind=0, skip_existing=False,
         single_thread_blas=False, just_copy_missing=False):
    if just_copy_missing:
        cpus = 1
    print(
        f"running on {cpus} cpus, processing {sdir}, writing into {tdir}, {'including' if not skip_target_only else 'skipping'} targets, starting from file #{skip_to_file_ind},  {'overwritting' if not skip_existing else 'skipping'} targets that already exist in destination")
    assert os.path.exists(sdir), f"Source directory '{sdir}' doesn't exist"
    if tdir is not None:
        if not os.path.exists(tdir):
            print(f"CREATING THE OUTPUT DIRECTORY {tdir}")
            os.mkdir(tdir)
        else:
            warnings.warn(
                f'TrainingDataWriter: target directory {tdir} already exists, gonna blindly overwrite (or add) files in that directory')

    files = glob.glob(os.path.join(sdir, '*.off'))
    if len(files) == 0:
        files = glob.glob(os.path.join(sdir, '*.obj'))
    assert len(files) > 0
    files.sort()
    jfile = "data.json"
    data_file = os.path.join(sdir, jfile)
    if os.path.isfile(data_file):
        cf = SourceToTargetsFile.load(data_file)
        if cf is None:
            with open(data_file) as file:
                data = json.load(file)
                cf = SourceToTargetsFile.SourceToTargetsFile(data['pairs'], files)
        pairs = cf.get_pair_names()
    else:
        pairs = None
    if skip_target_only:
        assert cf is not None, "if you specify skip_target_only = True, you need to have a data.json file to read which are targets from"
        only_target = cf.get_only_target_indices()
    else:
        only_target = {}

    t = TrainingDataWriter(files, tdir, sdir, file_inds_to_not_process=only_target, cpus=cpus, report_times=verbose,
                           start_file_ind=skip_to_file_ind, skip_existing=skip_existing,
                           single_thread_blas=single_thread_blas, just_copy_missing=just_copy_missing, pairs=pairs)
    t.run()
    print("xxxxxxx ...NOW MAIN IS REALLY EXITING :P (LAST PRINT BEFORE RETURNING) xxxxxxxxx")


def main_from_args():
    # if len(sys.argv) == 1:
    #     sys.argv.append('C:/final_for_star')
    #     sys.argv.append('-c')
    #     sys.argv.append('2')

    print("starting...")
    # dir_name = 'data/10k_surface_patches'

    import argparse

    dcpu = min(mp.cpu_count() - 2, 24)
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', default='data/faust')
    parser.add_argument("--skipexisting", help="dont process and write meshes that already exist in target",
                        action="store_true")
    parser.add_argument('-v', "--verbose", help="verbose reporting of timings", action="store_true")
    parser.add_argument('-s', "--save_dir",
                        help=f"destination dir to write to (content may be overwritten). Defaults to <input_dir>_processed",
                        type=str)
    parser.add_argument('-c', "--cpus",
                        help=f"number of cpus to use (default on this machine is {dcpu}, out of {mp.cpu_count()})",
                        type=int, default=dcpu)
    parser.add_argument('-ff', '--fastforward',
                        help='start from the given index of the file (sorted by lexigoraphical order)', type=int,
                        default=0)
    parser.add_argument("--skiptargets",
                        help="If specified,  don't perform analysis (like laplacian) on meshes that only act as target according to the json file. This option exists for speed considerations in cases where you know that a mesh will NEVER be used as a source.",
                        action="store_true")
    parser.add_argument("--singlethreadblas",
                        help="If specified,  run blas in single thread",
                        action="store_true")
    parser.add_argument("--copy_missing", help="If specified,  run blas in single thread",
                        action="store_true")
    parser.add_argument("--make_fake_json", help="create a json file that maps each mesh to itself",
                        action="store_true")

    args = parser.parse_args()
    verbose = args.verbose
    cpus = args.cpus
    dir_to_process = args.dir
    ff = args.fastforward
    if args.save_dir is None:
        dir_to_write_to = dir_to_process
        if dir_to_write_to[-1] == '/' or dir_to_write_to[-1] == '\\':
            dir_to_write_to = dir_to_write_to[0:-1]
        dir_to_write_to += "_processed/"
    else:
        dir_to_write_to = args.save_dir

    print(args)
    main(dir_to_process, dir_to_write_to, skip_target_only=args.skiptargets, cpus=cpus, verbose=verbose,
         skip_to_file_ind=ff, skip_existing=args.skipexisting, single_thread_blas=args.singlethreadblas,
         just_copy_missing=args.copy_missing)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('/home/groueix/db_test')
    main_from_args()
