# %%
import os
os.environ['R_HOME'] = r'C:\Programming\R\R-4.4.2'
import rpy2
print(rpy2.__version__)

# %%
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import numpy as np

# %%
print(r['R.version.string'][0])
print(rpy2.__version__)
# %%

from rpy2.robjects import r
import rpy2.robjects as robjects
r.source('test.R')
my_add = robjects.globalenv['my_add']
my_add(1, 2)

# %%

# %%
import os
os.environ['R_HOME'] = r'C:\Programming\R\R-4.4.2'

import rpy2
print(rpy2.__version__)

# %%
from rpy2.robjects import r
import rpy2.robjects as robjects

# cusum load
r_path = os.path.abspath(r'C:\Programming\Github\ADD\Cusum')
r.source(r_path + '\cusum_fitting_share.R')
# %%

print(r.ls())
# %%
start_point = r['start.point']
time_labels = r['time.labels'] 
print(start_point)
print(time_labels[start_point])

# %%
