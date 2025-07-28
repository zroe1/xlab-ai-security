<img width="1311" height="152" alt="Screenshot 2025-07-26 at 12 04 07 AM" src="https://github.com/user-attachments/assets/e05f61c7-998d-41e4-b148-5365936bfc5a" />
<br></br>
<p align="center">
<a href="https://test.pypi.org/project/xlab-security/"><img alt="link to PyPi" src="https://img.shields.io/badge/PyPI version-0.18-brightgreen"></a>
<a href="https://xlabaisecurity.com/"><img alt="link to PyPi" src="https://img.shields.io/badge/website%20build-passing-brightgreen"></a>

</p> 



# Overview

Course link: https://xlabaisecurity.com/

test pypi link for python package: https://test.pypi.org/project/xlab-security/

final pypi link for python package: https://pypi.org/project/xlab-security/

install testing version of package:
```
pip install --index-url https://test.pypi.org/simple/ xlab-security
```

Install production version of package:

```
pip install xlab-security
```

To import the package simply:

```
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
