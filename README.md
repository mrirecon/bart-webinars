# BART Webinars 

This repository contains information and materials for the [BART](http://mrirecon.github.io/bart) webinars:

https://mrirecon.github.io/bart/webinars.html

## First Webinar: June 1 and 2, 8-10am PST
Webinar Recordings: [Click here](https://www.youtube.com/playlist?list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY)

### Schedule: Day 1
- Introduction [[Recording]](https://www.youtube.com/watch?v=IPXhgy4Tymo&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=2&t=0s)

- Part 1:
  - [[Recording]](https://www.youtube.com/watch?v=iPujHGoyjMA&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=3&t=0s) Where to find docs, examples, and help
      (links: [Website](https://mrirecon.github.io/bart/) + [README](https://github.com/mrirecon/bart/blob/master/README) + [Documentation](https://github.com/mrirecon/bart/tree/master/doc) + [Tutorials](https://github.com/mrirecon/bart-workshop) + [Mailing List](https://lists.eecs.berkeley.edu/sympa/info/mrirecon))
  - Discussion of file format and dimensions
      (links: [README: Data Format](https://github.com/mrirecon/bart/blob/master/README#L333) + [Documentation: Dimensions](https://github.com/mrirecon/bart/blob/master/doc/dimensions-and-strides.txt) + [Source Code: Predefined Dimensions](https://github.com/mrirecon/bart/blob/master/src/misc/mri.h#L15))
       
- Part 2:
  - [[Recording]](https://www.youtube.com/watch?v=FPCTCljBdA4&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=4&t=0s) Working with CLI tools and Matlab/Python wrappers [Jupyter Notebook](./cli_python_matlab_intro.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-webinars/master?filepath=cli_python_matlab_intro.ipynb)
  - [[Recording]](https://www.youtube.com/watch?v=vBYI7rloK-A&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=5&t=0s) Data preprocessing [Jupyter Notebook](./webinar-preprocessing.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-webinars/master?filepath=webinar-preprocessing.ipynb)

- Part 3:  
  - [[Recording]](https://www.youtube.com/watch?v=YN9dlpw86qg&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=6&t=1s) Compressed Sensing and non-Cartesian MRI reconstruction [Jupyter Notebook](./day1_advanced_recon.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-webinars/master?filepath=day1_advanced_recon.ipynb)
  - [[Recording]](https://www.youtube.com/watch?v=BoKtya6qd1c&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=7&t=0s) GRASP-like MRI reconstruction (same Jupyter Notebook as above)

- [[Recording]](https://www.youtube.com/watch?v=nEeCcBQb2r8&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=8&t=0s) Q&A and Conclusion

### Schedule: Day 2
- [[Recording]](https://www.youtube.com/watch?v=n4Vm_TPgWEE&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=9&t=0s) Introduction  
- [[Recording]](https://www.youtube.com/watch?v=5CMkzyjmdI0&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=10&t=0s) Part 1: Intro to the C-programming library [Slides](./Backend_Libraries_in_Bart.pdf)
 - [[Recording]](https://www.youtube.com/watch?v=h8r2GJCRG6A&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=11&t=0s) Part 2: Set up a build environment, compile, and run (links: [README](https://github.com/mrirecon/bart/blob/master/README) + [Documentation: Building](https://github.com/mrirecon/bart/blob/master/doc/building.txt) + [Documentation: Debugging](https://github.com/mrirecon/bart/blob/master/doc/debugging.txt))
- [[Recording]](https://www.youtube.com/watch?v=TiL-mS7UN_Y&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=12&t=0s) Part 3: Add command-line paramter to existing tool  [Jupyter Notebook](./day2_add_cmdline_parameter.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-webinars/master?filepath=day2_add_cmdline_parameter.ipynb)
- [[Recording]](https://www.youtube.com/watch?v=jJklLdcd1rs&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=13&t=0s) Part 4: Build a basic tool and system test [Jupyter Notebook](./day2_build_new_tool.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-webinars/master?filepath=day2_build_new_tool.ipynb)  
- [[Recording]](https://www.youtube.com/watch?v=qxF5y_qPfeg&list=PLDaugjrMfSRF0WhQ0nbcH4zeHWZPboGDY&index=14&t=0s) Q&A and Conclusion

## Acknowledgements
This work is supported by NIH Grant U24EB029240-01
