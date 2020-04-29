# Pre-requisites

- Clone the project
- Install Anaconda

Run following command to create identical conda environment as mine. Lets use this to keep the same environment for everyone.

`conda env create -f environment.yml`

# How to setup a python virtual environment

- Download and install [Python 3.6.x](https://www.python.org/downloads/release/python-3610/)
- You can verify that you have Python 3.6 installed using the following command on a terminal/shell:

Linux:

```
python3.6 --version
```

- Create a `venv` using the following command:

```
python3.6 -m venv folder_name
```

- Activate the venv from the terminal(in the directory where venv folder resides):

```
source folder_name/bin/activate
```

- Update pip and setuptools:

```
pip install -U pip
pip install -U setuptools
```

- Check the list of installed packaged and their versions:

```
pip list
```

- Switch the directory of the terminal/shell to the repo where you have `requirements.txt` file and then install the requirements:

```
pip install -r requirements.txt
```

- You can reverify that everything is fine using:

```
pip list
```

- You can then use this venv with and IDE like `pycharm` or activate it if you are using text editors like `VS Code`

- Once you are done working with the venv deactivate using:

```
deactivate
```
