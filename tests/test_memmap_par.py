import numpy as np
import multiprocessing as mp


def worker(arr,beg,end, mm):
    # beg = (row,col) upper left corner
    # end = (row,col) lower right corner
    proc = mp.current_process()
    pid = proc.pid
    arr[beg[0]:end[0],beg[1]:end[1]] = pid
    mm[pid] = str(pid)


def main():
    minmax= mp.Manager().dict()
    fp = np.memmap('test_memmap.npy',  dtype='float32', mode='w+', shape=(100,100))
    p1 = mp.Process(target=worker, args=(fp,[0,0],[100,50],minmax))
    p1.start()
    p2 = mp.Process(target=worker, args=(fp,[0,50],[100,100],minmax))
    p2.start()
    print('end')
    print(minmax)

if __name__ == '__main__':
    main()
