<img width="1311" height="152" alt="Screenshot 2025-07-26 at 12 04 07 AM" src="https://github.com/user-attachments/assets/e05f61c7-998d-41e4-b148-5365936bfc5a" />
<br></br>
<p align="center">
<a href="https://test.pypi.org/project/xlab-security/"><img alt="link to PyPi" src="https://img.shields.io/badge/PyPI version-0.1.10-brightgreen"></a>
<a href="https://xlabaisecurity.com/"><img alt="link to PyPi" src="https://img.shields.io/badge/website%20build-passing-brightgreen"></a>

</p> 



## Overview

Welcome to the XLab AI Security Guide. **You can view the contents of the course at https://xlabaisecurity.com/**

This course covers a wide variety of topics in AI Security. For each topic we cover, there is a webpage describing the concept ([example here](https://xlabaisecurity.com/adversarial/cw/)). For most topic pages, the website will link to a series of coding excercises for hands-on experience with the concepts we describe. You will have the option to run the code you write either locally or in the cloud with Google Colab. 

This project is still in development. For now, most of the content in the [adversarial basis](https://xlabaisecurity.com/adversarial/introduction/) section has been completed. We are currently developing the [LLM jailbreaking](https://xlabaisecurity.com/jailbreaking/introduction/) sections and hope to be done in the next few weeks. 

If you want to get started using our guide, below are some useful links:

1. [The welcome page](https://xlabaisecurity.com/getting-started/welcome/): This explains what we mean by "AI security" and why we think this area of work is important.
2. [Prerequisites](https://xlabaisecurity.com/getting-started/prerequisites/): Before diving into the content of the course, make sure you read this page so you are aware of what we assume you know.
3. [Installation](https://xlabaisecurity.com/getting-started/set-up/): If you are not planning on running your code in Google Colab, you will have to install a few packages. This page provides instructions for how to install these packages, including [xlab-security](https://pypi.org/project/xlab-security/) which we developed internally.

## Development Details

In this monorepo, we include our Python package, the website, and the code for every pre-trained model we developed for this course. Note that all models are hosted on our [Hugging Face](https://huggingface.co/uchicago-xlab-ai-security) rather than our GitHub.

### Python Package

This repository contains the code for `xlab-security`, a Python package that we have developed for this course. Students use this package to import useful utility functions and run tests for the coding excercises they complete. For development purposes, we have uploaded the package to the testing version of the PyPI package manager. You can view the testing version of the package at [this link]( https://test.pypi.org/project/xlab-security/) (though we warn that we don't update the testing version frequently). 

To install the testing version of the package, you can run:
```
pip install --index-url https://test.pypi.org/simple/ xlab-security
```

The link to the production version of the Python package can be found [here](https://pypi.org/project/xlab-security/). To install the production version of the package, you can run:

```
pip install xlab-security
```

Once you have installed the package, you can import it within your Python file or Jupyter notebook by writing:

```python
import xlab
```

## For the web app

To run the app locally run:

```
cd ai-security-course
npm run dev
```


To build the app run:

```
rm -rf .next
npm run export
```

This will generate an `out` directory which can be placed on a web server.
