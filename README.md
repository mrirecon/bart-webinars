# BART Webinars 

This repository contains information and materials for the [BART](http://mrirecon.github.io/bart) webinars:

https://mrirecon.github.io/bart/webinars.html

## First Webinar: June 1 and 2, 8-10am PST
Local time: [Click here](https://www.timeanddate.com/worldclock/fixedtime.html?msg=BART+Webinar&iso=20200601T08&p1=791&ah=2)

### Schedule: Day 1
- Introduction

- Part 1:
  - Where to find docs, examples, and help
      (links: [Website](https://mrirecon.github.io/bart/) + [README](https://github.com/mrirecon/bart/blob/master/README) + [Documentation](https://github.com/mrirecon/bart/tree/master/doc) + [Tutorials](https://github.com/mrirecon/bart-workshop) + [Mailing List](https://lists.eecs.berkeley.edu/sympa/info/mrirecon))
  - Discussion of file format and dimensions
      (links: [README: Data Format](https://github.com/mrirecon/bart/blob/master/README#L333) + [Documentation: Dimensions](https://github.com/mrirecon/bart/blob/master/doc/dimensions-and-strides.txt) + [Source Code: Predefined Dimensions](https://github.com/mrirecon/bart/blob/master/src/misc/mri.h#L15))
       
- Part 2:
  - Working with CLI tools and Matlab/Python wrappers [Jupyter Notebook](./cli_python_matlab_intro.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-webinars/master?filepath=cli_python_matlab_intro.ipynb)
  - Data preprocessing [Jupyter Notebook](./webinar-preprocessing.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-webinars/master?filepath=webinar-preprocessing.ipynb)

- Part 3:  
  - Compressed Sensing and non-Cartesian MRI reconstruction [Jupyter Notebook](./day1_advanced_recon.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-webinars/master?filepath=day1_advanced_recon.ipynb)
  - GRASP-like MRI reconstruction (same Jupyter Notebook as above)

- Q&A and Conclusion

### Schedule: Day 2
- Introduction  
- Part 1: Intro to the C-programming library [Slides](./Backend_Libraries_in_Bart.pdf)
 - Part 2: Set up a build environment, compile, and run (links: [README](https://github.com/mrirecon/bart/blob/master/README) + [Documentation: Building](https://github.com/mrirecon/bart/blob/master/doc/building.txt) + [Documentation: Debugging](https://github.com/mrirecon/bart/blob/master/doc/debugging.txt))
- Part 3: Add command-line paramter to existing tool  
- Part 4: Build a basic tool and system test [Jupyter Notebook](./day2_build_new_tool.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-webinars/master?filepath=day2_build_new_tool.ipynb)  
- Q&A and Conclusion

## Acknowledgements
This work is supported by NIH Grant U24EB029240-01
