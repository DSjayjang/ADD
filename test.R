# my_script.R
my_add <- function(x, y) {
  return(x + y)
}

my_add(1,2)

# install.packages('reticulate')
library(reticulate)

setwd('C:\\Programming\\Github\\ADD')

pd <- import('pandas')
source_python('test.py')
