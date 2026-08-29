
# johnny-appleseed

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [How to use](#how-to-use)


## About

johnny-appleseed is a tool I created that exports the logic from a Decision Tree Classifier created with the scikit-learn library to another language. Additionally, I have found that this tool can also help with visualizing and explaining the models that it exports. For convenience, some languages already exist as a preset, such as C, Python, Java, Ruby, and more. Otherwise, custom languages can be used via a dictionary parameter.

### 1. Clone repository

```shell
git clone https://github.com/carsonkoball/johnny-appleseed.git
cd amon-hen
```

### 2. Create virtual environment (recommended)

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```
**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```
**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install core dependencies

```shell
pip install -r requirements.txt
pip install -e .
```

**Example notebook (optional):**
```shell
cd example
pip install -r requirements.txt
```

##  How to Use
First, import the module:
```
from johnny_appleseed.tree_exporter import TreeExporter
```
Assuming you already have a trained classifier, ``clf``, you can instantiate a ``TreeExporter`` class:
```
te = TreeExporter(clf)
```

Then, you can export the model to another language, such as ``C`` and put the results in a file, such as ``output.txt``:
```
te.export(
	language='C',
	output_file_name='output.txt'
)
```

If you would like to change the names of features or classes (such as for variable names), you can easily customize the mapping with dictionaries for the ``feature_map`` and ``class_map`` parameters. For example, to change ``Feature 1`` to ``feature_one`` and ``Class 1`` to ``class_one``, you can map something like:
```
te.export(
	language='C',
	feature_map={
		'Feature 1': 'feature_one'
	},
	class_map={
		'Class 1': 'class_one'
	},
	output_file_name='output.txt'
)
```

You can easily see all of the available language presets with the ``get_languages()`` function:
```
te.get_languages()
```

Or you can view the properties of a language preset with the ``get_language_preset()`` function, such as with ``C``:
```
te.get_language_preset('C')
```

If none of the language presets fit your need, you can create your own by defining the language properties with a dictionary for the ``language`` parameter instead of a string:
```
te.export(
	language={
		'indentation': ' ',
		'if': 'if the ',
		'if_end': '',
		'condition': ' feature is less than or equal to ',
		'then': ',',
		'else': 'otherwise,',
		'else_end': '',
		'set': ' = ',
		'variable_operator': '',
		'feature_name_prefix': '',
		'feature_name_suffix': '',
		'result_prefix': 'the sample is ',
		'result_suffix': '.',
		'threshold_formatter': '.4f'
	},
	output_file_name='output.txt'
)
```

Example outputs can be seen in [/example/tree_exporter_example.ipynb](/example/tree_exporter_example.ipynb).