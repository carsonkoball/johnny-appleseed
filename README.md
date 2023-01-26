# johnny-appleseed

## Table of Contents
- [About](#about)
- [Installation](#installation)
- [How to use](#how-to-use)


## About
johnny-appleseed is a tool I created that exports the logic from a Decision Tree Classifier created with the scikit-learn library to another language. Additionally, I have found that this tool can also help with visualizing and explaining the models that it exports. For convenience, some languages already exist as a preset, such as C, Python, Java, Ruby, and more. Otherwise, custom languages can be used via a dictionary parameter.

## Installation
### Dependencies
- Python
- scikit-learn
- NumPy
- json
- pkg_resources

### User installation
#TODO

##  How to Use
First, import the module:
```
from johnny_appleseed import TreeExporter
```
Assuming you already have a trained classifier, ``clf``, you can initialize a ``TreeExporter`` class:
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

Example outputs can be seen in the ``/tests/test_TreeExporter.ipynb`` file.
