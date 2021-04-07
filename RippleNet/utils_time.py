import time
import os



class TimeTickTock():
    def __init__(self):
        self.reset()
        self._enable=True
        self.setting()

    def tick(self):
        self.tic = time.time()

    def tock(self, message=None,layer=0 ):
        if self._enable:
            self.messages.append(message)
            self.times.append(time.time()-self.tic)
        self.tic = time.time()

    def reset(self):
        self.times=[]
        self.messages=[]
        self.tic = time.time()

    def show(self, demical=None, show_messages=None, separate=None, show_total_time=True):
        if not self._enable:
            return
        #print(locals())
        #self.updata_settings(locals())
        if separate is None:
            separate = self._separate
        if show_messages is None:
            show_messages = self._show_messages
        if demical is None:
            demical = self._demical

        cout_string=''
        if show_messages:
            for  ti,mess in zip(self.times,self.messages):
                if mess is None:
                    cout_string+=str(round(ti,demical))+ separate
                else:
                    cout_string+=mess+':'+ str(round(ti,demical)) + separate

        else:
            for ti in self.times:
                cout_string+=str(round(ti,demical))+separate
        if show_total_time:
            cout_string+='total_time='+str(round(sum(self.times),demical))
        print(cout_string)

    def enable(self, enable=True):
        self._enable=enable

    def setting(self, separate=', ',demical=3,show_messages=True, show_total_time=True):
        self._separate = separate
        self._demical = demical
        self._show_messages = show_messages
        self._show_total_time=show_total_time
