## INSTALLATION SCRIPT TO GET GRETA, CAUSACT,
## and TENSORFLOW ALL WORKING TOGETHER HAPPILY

## NOTE:  Run each line one at a time using CTRL+ENTER.
##        Await completion of one line
##        before running the next.
##        If prompted to "Restart R", say YES.

#### STEP 0:  Restart R in a Clean Session
#### use RStudio menu:  SESSION -> RESTART R

#### STEP 1: INSTALL PACKAGES WITH PYTHON DEPENDENCIES
install.packages("reticulate",dependencies = TRUE)
install.packages("greta",dependencies = TRUE)
install.packages("causact",dependencies = TRUE)

#### STEP 2: INSTALL & UPDATE MINICONDA SO R CAN FIND PYTHON 
## install miniconda in default location if possible
condaInstall = try(reticulate::install_miniconda())
condaPath = try(reticulate::miniconda_path())
## if ERROR is due to a previous installation, then ignore the error.
## if install fails due to a space in your path, then uncomment
## the below two lines and run them.  
## condaPath = file.path("/", "miniconda")
## reticulate::install_miniconda(path = condaPath,force = TRUE)}

#### STEP 3: Add environment variable so that 
#### reticulate does not attempt to automatically configure
#### a python environment for you
rEnvPath = file.path("~", ".Renviron")
envLines = c()  ## init blank lines
if (file.exists(rEnvPath)) {
  envLines = readLines(rEnvPath)# get rProfile
}
## add new line to bottom of file
newLine = 'RETICULATE_AUTOCONFIGURE = "FALSE"'
envLines = c(envLines, newLine)
writeLines(envLines, rEnvPath)
## also set line for current session
Sys.setenv(RETICULATE_AUTOCONFIGURE = FALSE)

#### STEP 4:  Update "r-reticulate" CONDA ENVIRONMENT
####          FOR TENSORFLOW
## Install the specific versions of modules
## for the TensorFlow installation via CONDA.
## these next lines may take a few minutes to execute
reticulate::conda_remove("r-reticulate")  #start clean
reticulate::py_config()  # initiate basic r-reticulate config -- ignore any error here
## install other packages and downgrade numpy
reticulate::conda_install(envname = "r-reticulate",
                          packages =
                            c(
                              "python=3.7",
                              "tensorflow=1.14",
                              "pyyaml",
                              "requests",
                              "Pillow",
                              "pip",
                              "numpy=1.16",
                              "h5py=2.8",
                              "tensorflow-probability=0.7"
                            ))

#### STEP 4:  TEST THE INSTALLATION - must restart r
##  **** USE MENU:   SESSION -> RESTART R
library(greta)  ## should work now if you restarted R.. takes a minute
library(causact)
graph = dag_create() %>%
  dag_node("Normal RV",
           rhs =normal(0,10))
graph %>% dag_render()  ## see oval
drawsDF = graph %>% dag_greta() ## see "running X chains..."
drawsDF %>% dagp_plot(densityPlot = TRUE)  ## see plot

#### STEP 5:  RESTART R AND TRY STEP 4 JUST TO ENSURE
#### ALL IS WELL
#### USE MENU:  SESSION -> RESTART R
#### CONGRATS IF IT WORKS.  