from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import str
from traits.api import HasTraits, String
import pickle as pickle
import sys, traceback, os.path

class HasTraitsPersistent(HasTraits):
    "custom Traits class with methods for loading/saving state"

    settings_id = String('state', transient = True)
    
    def load_settings(self):
        try:
            sfilename = 'config/%s.pkl'%self.settings_id
            sfilename = os.path.join(os.path.split(__file__)[0], sfilename)
            with open(sfilename, 'rb') as sfile:
                state = pickle.load(sfile)
                state.pop('__traits_version__')
                print("load settings", self.settings_id)
                self.trait_set(trait_change_notify = True, **state)
        except Exception as e:
            print("error loading settings", end=' ')
            exc_type, exc_value, exc_tb = sys.exc_info()
            info = str(e)
            log = '\n'.join(traceback.format_exception(exc_type, exc_value, exc_tb)[2:])
            print(log)
            print(str(self.__class__))
            print(e)
            print()

    def dump_settings(self):
        sfilename = 'config/%s.pkl'%self.settings_id
        sfilename = os.path.join(os.path.split(__file__)[0], sfilename)
        with open(sfilename, 'wb') as sfile:
            state = self.__getstate__()
            for key in list(state):
                if key.startswith('_') and key is not '__traits_version__':
                    state.pop(key)
            pickle.dump(state, sfile)
        print("saved settings", self.settings_id)
