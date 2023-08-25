#!/usr/bin/env python

from tasks import *
from glob import glob
from os.path import basename

for filename in glob("tasks/*.py"):
    taskname = basename(filename).replace(".py", "")
    print (taskname)
    task = __import__(f"tasks.{taskname}").__dict__[taskname]
    task.decode()
    
