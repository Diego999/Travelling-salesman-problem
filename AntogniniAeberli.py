#!/usr/bin/env python
"""usage: AntogniniAeberli.py [options] [params] [cityfile]

options:
-h, --help Show this help
-n, --no-gui create AST pdf

params:
-m VALUE, --maxtime=VALUE  Max execution time of genetic algorithm.
                           Negative values for infinite. Default: 0

(c) 2014 by Diego Antognini and Marco Aeberli")
"""

import sys
import getopt
import os

def usage():
    """Prints the module how to usage instructions to the console"
    """
    print(__doc__)

def get_argv_params():
    """Recuperates the arguments from the command line
    """
    try:
        opts = getopt.getopt(
            sys.argv[1:],
            "hnm:",
            ["help","no-gui","maxtime="] )[0]
    except getopt.GetoptError:
        usage()
        print("Wrong options or params.")
        exit(2)
        
    gui = True
    maxtime = 0
    filename = None
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            exit()
        elif opt in ("-n", "--no-gui"):
            gui = False
        elif opt in ("-m", "--maxtime"):
            maxtime = arg
            
    filename = sys.argv[-1]
    if not  os.path.exists(filename) or len(sys.argv) <= 1:
        usage()
        print("invalid city file: %s" % filename)
        exit(2)

    return (gui, maxtime, filename)

if __name__ == "__main__":
    (GUI, MAXTIME, FILENAME) = get_argv_params()
    
    print("args gui: %s maxtime: %s filename: %s" % (GUI, MAXTIME, FILENAME))
