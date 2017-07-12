#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy
from deepgtav.client import Client

import argparse
import time
import cv2
import sys
import numpy as np

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QRect
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap

def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)


# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug

class Worker(QObject):
    """
    Must derive from QObject in order to emit signals, connect slots to other signals, and operate in a QThread.
    """

    sig_step = pyqtSignal(int, str)  # worker id, step description: emitted every step through work() loop
    sig_done = pyqtSignal(int)  # worker id: emitted at end of work()
    sig_msg = pyqtSignal(str)  # message to be shown to user
    sig_image = pyqtSignal(list)

    def __init__(self, id: int, args):
        super().__init__()
        self.__id = id
        self.__abort = False
        self.args = args

    @pyqtSlot()
    def work(self):
        """
        Pretend this worker method does work that takes a long time. During this time, the thread's
        event loop is blocked, except if the application's processEvents() is called: this gives every
        thread (incl. main) a chance to process events, which in this sample means processing signals
        received from GUI (such as abort).
        """
        thread_name = QThread.currentThread().objectName()
        thread_id = int(QThread.currentThreadId())  # cast to int() is necessary
        self.sig_msg.emit('Running worker #{} from thread "{}" (#{})'.format(self.__id, thread_name, thread_id))

                     
        # Creates a new connection to DeepGTAV using the specified ip and port. 
        # If desired, a dataset path and compression level can be set to store in memory all the data received in a gziped pickle file.
        # We don't want to save a dataset in this case
        self.client = Client(ip=self.args.host, port=self.args.port)
        # self.client = Client(ip="127.0.0.1", port=8000)
        
        # We set the scenario to be in manual driving, and everything else random (time, weather and location). 
        # See deepgtav/messages.py to see what options are supported
        scenario = Scenario(drivingMode=-1) #manual driving
        
        # Send the Start request to DeepGTAV. Dataset is set as default, we only receive frames at 10Hz (320, 160)
        self.client.sendMessage(Start(scenario=scenario))
        
        # Dummy agent
        model = Model()

        # Start listening for messages coming from DeepGTAV. We do it for 80 hours
        stoptime = time.time() + 80*3600
        while (time.time() < stoptime and (not self.__abort)):
            # We receive a message as a Python dictionary
            app.processEvents()
            message = self.client.recvMessage() 
            
                
            # The frame is a numpy array that can we pass through a CNN for example     
            image = frame2numpy(message['frame'], (320,160))
            commands = model.run(image)
            self.sig_step.emit(self.__id, 'step ' + str(time.time()))
            self.sig_image.emit(image.tolist())
            # We send the commands predicted by the agent back to DeepGTAV to control the vehicle
            self.client.sendMessage(Commands(commands[0], commands[1], commands[2]))
        
            
        # We tell DeepGTAV to stop
        self.client.sendMessage(Stop())
        self.client.close()
                
        self.sig_done.emit(self.__id)

    def abort(self):
        self.sig_msg.emit('Worker #{} notified to abort'.format(self.__id))
        self.__abort = True
        

class MyWidget(QWidget):
    NUM_THREADS = 1

    # sig_start = pyqtSignal()  # needed only due to PyCharm debugger bug (!)
    sig_abort_worker = pyqtSignal()

    def __init__(self, args):
        super().__init__()

        self.args = args
        
        self.setWindowTitle("VPilot Drive GUI")
        form_layout = QVBoxLayout()
        self.setLayout(form_layout)
        self.resize(400, 800)

                
        self.button_start = QPushButton()
        self.button_start.clicked.connect(self.start_thread)
        self.button_start.setText("Start")
        form_layout.addWidget(self.button_start)

        self.button_stop = QPushButton()
        self.button_stop.clicked.connect(self.abort_worker)
        self.button_stop.setText("Stop")
        self.button_stop.setDisabled(True)
        form_layout.addWidget(self.button_stop)

        # self.log = QTextEdit()
        # form_layout.addWidget(self.log)

        self.label = QLabel(self)
        # self.label.setGeometry(QRect(20, 20, 320, 160))
        self.label.resize(320, 160)
        
        # self.progress = QTextEdit()
        # form_layout.addWidget(self.progress)

        QThread.currentThread().setObjectName('main')  # threads can be named, useful for log output
        self.__worker_done = None
        self.__thread = None

    def start_thread(self):
        # self.log.append('starting thread')
        self.button_start.setDisabled(True)
        self.button_stop.setEnabled(True)

        self.__worker_done = 0
        self.__thread = []
        # for idx in range(self.NUM_THREADS):
        idx = 0
        worker = Worker(idx, self.args)
        thread = QThread()
        thread.setObjectName('thread_' + str(idx))
        self.__thread.append((thread, worker))  # need to store worker too otherwise will be gc'd
        worker.moveToThread(thread)

        # get progress messages from worker:
        worker.sig_step.connect(self.on_worker_step)
        worker.sig_done.connect(self.on_worker_done)
        # worker.sig_msg.connect(self.log.append)
        worker.sig_image.connect(self.on_image)

        # control worker:
        self.sig_abort_worker.connect(worker.abort)

        # get read to start worker:
        # self.sig_start.connect(worker.work)  # needed due to PyCharm debugger bug (!); comment out next line
        thread.started.connect(worker.work)
        thread.start()  # this will emit 'started' and start thread's event loop

        # self.sig_start.emit()  # needed due to PyCharm debugger bug (!)

    @pyqtSlot(int, str)
    def on_worker_step(self, worker_id: int, data: str):
        # self.log.append('Worker #{}: {}'.format(worker_id, data))
        # self.progress.append('{}: {}'.format(worker_id, data))
        pass

    @pyqtSlot(int)
    def on_worker_done(self, worker_id):
        # self.log.append('worker #{} done'.format(worker_id))
        # self.progress.append('-- Worker {} DONE'.format(worker_id))
        # self.log.append('No more workers active')
        self.button_start.setEnabled(True)
        self.button_stop.setDisabled(True)
       
    @pyqtSlot(list)
    def on_image(self, image):
        image_np = np.array(image).astype(np.uint8)
        height, width, channel = image_np.shape
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        qImg = QImage(image_np.data, width, height, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qImg)
        self.label.setPixmap(pix)

    @pyqtSlot()
    def abort_worker(self):
        self.sig_abort_worker.emit()
        # self.log.append('Asking each worker to abort')
        thread = self.__thread[0][0]  
        worker = self.__thread[0][1] 

        thread.quit()  # this will quit **as soon as thread event loop unblocks**
        thread.wait()  # <- so you need to wait for it to *actually* quit

        # even though threads have exited, there may still be messages on the main thread's
        # queue (messages that threads emitted before the abort):
        # self.log.append('All threads exited')



class Model:
    def run(self,frame):
        return [1.0, 0.0, 0.0] # throttle, brake, steering

# Controls the DeepGTAV vehicle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    args = parser.parse_args()

    app = QApplication([])

    form = MyWidget(args)
    form.show()

    sys.exit(app.exec_())
  
