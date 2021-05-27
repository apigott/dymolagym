
from dymola.dymola_interface import DymolaInterface
import os
from multiprocess import Process, Array, Queue

def call_dymola(dymola, stop):
    result = dymola.simulateExtendedModel('OpenIPSL.Examples.IEEE9.IEEE_9_Base_Case_OL', stopTime=stop, finalNames=['integrator'])[1]
    queue.put(result)

if __name__=="__main__":
    dymola = DymolaInterface()
    dymola.ExecuteCommand("Advanced.Define.DAEsolver = true")

    # load libraries
    loaded = []
    for lib in ["../../OpenIPSL-1.5.0/OpenIPSL/package.mo"]: # all paths relative to the cwd
        loaded += [dymola.openModel(lib, changeDirectory=False)]

    # if not False in loaded:
    #     logger.debug("Successfully loaded all libraries.")
    # else:
    #     logger.error("Dymola could not find all models.")

    if not os.path.isdir('temp_dir'):
        os.mkdir('temp_dir')
        temp_dir = os.path.join(os.getcwd(), "temp_dir")
        dymola.cd('temp_dir')

    queue = Queue()

    p = Process(target=call_dymola, args=[dymola, 1])
    p.start()
    print(queue.get())
    p.join(timeout=4)
    p.terminate()
