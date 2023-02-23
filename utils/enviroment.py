import os
from LookGenerator.config.config import WEIGHTS_URL, WEIGHTS_DIR



def load_weights(): 
    if os.path.exists(WEIGHTS_DIR):
         return True
    
    else: 

        #Download files to WEITHGS_DIR from WEIGHTS_UTL
        pass