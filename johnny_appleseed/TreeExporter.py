import json
import warnings

import numpy as np
import pkg_resources
from sklearn.tree import DecisionTreeClassifier


class TreeExporter():
    """Tool for exporting scikit-learn Decision Tree Classifiers
    to an if-else structure in a language of choice.
    
    
    Parameters
    ----------
    Tree : 
        The Decision Tree Classifier to export.
    """

    def __init__(self, Tree):
        # check to see if user inputs a scikit-learn Decision Tree Classifier
        if not isinstance(Tree, DecisionTreeClassifier):
            raise TypeError('Input is not a scikit-learn DecisionTreeClassifier() object.')

        # number of nodes that comprise the classifier
        self.n_nodes = Tree.tree_.node_count

        # id of left children of each node
        self.children_left = Tree.tree_.children_left

        # id of left children of each node
        self.children_right = Tree.tree_.children_right

        # features used for splitting each node
        self.features = Tree.tree_.feature

        # conditional threshold value for splitting each node
        self.thresholds = Tree.tree_.threshold

        # each feature seen by the classifier in the fitting phase
        if hasattr(Tree.tree_, 'feature_names_in'):
            # classifier was trained with feature names explicitly
            self.feature_names = Tree.feature_names_in_
        else:
            # classifier was NOT trained with feature names explicitly - numbers will be used instead
            warnings.warn('tree was not fitted with feature names - using numbers instead', RuntimeWarning)
            self.feature_names = list(range(Tree.n_features_in_))

        # leaf status for each node in classifier 1 if leaf 0 if subroot
        self.is_leaf = [1 if self.children_left[node] == self.children_right[node] else 0 for node in range(self.n_nodes)]

        # each class seen by the classifier in the fitting phase
        self.classes = [Tree.classes_[np.argmax(Tree.tree_.value[i])] for i in range(self.n_nodes)]
        
        # the exported tree stored as a string
        self.exported_tree = ''


    def __writer(self, language_dict, feature_map, class_map, output_file_name=''):
        """The main writer for the Decision Tree Classifier code.
        ----------
        language_dict : dictionary
            The dictionary containing properties of the desired language.
        feature_map : dictionary
            A dictionary that maps the feature names found in the tree
            to the desired feature names in the exported language.
        class_map : dictionary
            A dictionary that maps the class names found in the tree
            to the desired class names in the exported language.
        output_file_name : string, optional
            The file name to which the tree will be exported to as text.
            If the file doesn't exist already, it will be created. If the
            file already exists, it will be overwritten. If a file name
            is not specified, the tree will not be exported to a file.

        Returns
        -------
        self.exported_tree : string
             The exported tree stored as a string.
        """
        
        # the tree itself
        self.__tree_writer(language_dict, feature_map, class_map)
        
        if output_file_name != '':
            # file specified, export to said file
            with open(output_file_name, 'w') as file:
                file.write(self.exported_tree)

        return self.exported_tree
      
    def __writer_leaf(self, language_dict, class_map, node):
        """Writer for a decision tree leaf node.
        ----------
        language_dict : dictionary
            The dictionary containing properties of the desired language.
        class_map : dictionary
            A dictionary that maps the class names found in the tree
            to the desired class names in the exported language.
        node : int
            The current node to evaluate.

        Returns
        -------
        Nothing
        """
        
        self.exported_tree += language_dict['result_prefix']
        if self.classes[node] in class_map:
            # if the class name is found in the class map, use it instead
            self.exported_tree += str(class_map[self.classes[node]])
        else:
            # otherwise, just use the default class name found in the tree
            self.exported_tree += str(self.classes[node])
        self.exported_tree += language_dict['result_suffix']
        self.exported_tree += '\n'
        
    def __writer_split(self, language_dict, feature_map, node):
        """Writer for a decision tree split node.
        ----------
        language_dict : dictionary
            The dictionary containing properties of the desired language.
        feature_map : dictionary
            A dictionary that maps the feature names found in the tree
            to the desired feature names in the exported language.
        node : int
            The current node to evaluate.

        Returns
        -------
        Nothing
        """
        
        # if structure
        self.exported_tree += language_dict['if']
        self.exported_tree += language_dict['variable_operator']
        self.exported_tree += language_dict['feature_name_prefix']

        if self.feature_names[self.features[node]] in feature_map:
            # if the feature name is found in the feature map, use it instead
            self.exported_tree += feature_map[self.feature_names[self.features[node]]]
        else:
            # otherwise, just use the default feature name found in the tree
            self.exported_tree += self.feature_names[self.features[node]]

        self.exported_tree += language_dict['feature_name_suffix']
        self.exported_tree += language_dict['condition']
        self.exported_tree += str(format(self.thresholds[node], language_dict['threshold_formatter']))
        self.exported_tree += language_dict['then']
        
    def __tree_writer(self, language_dict, feature_map, class_map, node=0, indentation_count=0):
        """Performs a preorder traversal of the Decision Tree Classifier
        and writes the result to the self.exported_tree string.
        ----------
        language_dict : dictionary
            The dictionary containing properties of the desired language.
        feature_map : dictionary
            A dictionary that maps the feature names found in the tree
            to the desired feature names in the exported language.
        class_map : dictionary
            A dictionary that maps the class names found in the tree
            to the desired class names in the exported language.
        node : int
            The current node to evaluate.
        indentation_count : int
            The current indentation level.

        Returns
        -------
        Nothing
        """
        if (node != self.n_nodes):
            if self.is_leaf[node] == 1:
                # leaf node
                self.exported_tree += language_dict['indentation'] * indentation_count
                self.__writer_leaf(language_dict, class_map, node)
                
                return
            else:
                # split node
                self.exported_tree += language_dict['indentation'] * indentation_count
                self.__writer_split(language_dict, feature_map, node)
                self.exported_tree += '\n'
                                  
            # traverse left down the tree
            self.__tree_writer(language_dict, feature_map, class_map, self.children_left[node], indentation_count+1)
            if language_dict['if_end'] != '':
                self.exported_tree += language_dict['indentation'] * indentation_count
                self.exported_tree += language_dict['if_end']
                self.exported_tree += '\n'

            # insert else on return
            self.exported_tree += language_dict['indentation'] * indentation_count
            self.exported_tree += language_dict['else']
            self.exported_tree += '\n'
            
            # traverse right down the tree
            self.__tree_writer(language_dict, feature_map, class_map, self.children_right[node], indentation_count+1)
            if language_dict['else_end'] != '':
                self.exported_tree += language_dict['indentation'] * indentation_count
                self.exported_tree += language_dict['else_end']
                self.exported_tree += '\n'
            
    def __get_language_dict(self, language):
        """Retrieve language properties from presets of languages found
        in language_dicts.dat.
        ----------
        language : string
            The language whose properties will be retrieved from the
            language_dicts.dat file.

        Returns
        -------
        language_dict : dictionary
             The dictionary containing properties of the desired language.
        """
        
        language_dict = {}
        
        # reading the data from the file
        stream = pkg_resources.resource_filename('johnny_appleseed', 'data/language_dicts.dat')

        with open(stream) as f:
            data = f.read()

        # reconstructing the data as a dictionary of dictionaries
        language_dicts = json.loads(data)

        # finding only the dictionary of the language we want
        for l in language_dicts['languages']:
            if l['name'] == language:
                language_dict = l['properties']
            
        # language not found
        return language_dict

    def export(self, language, feature_map={}, class_map={}, output_file_name=''):
        """Export the Decision Tree Classifier to the language of choice.
        ----------
        language : string or dictionary
            The language that the Decision Tree Classifier will be
            exported to. If string, some presets are given in language_dicts.dat,
            otherwise the user will define the details of their language
            with a dictionary.
        feature_map : dictionary, optional
            A dictionary that maps the feature names found in the tree
            to the desired feature names in the exported language.
        class_map : dictionary, optional
            A dictionary that maps the class names found in the tree
            to the desired class names in the exported language.
        output_file_name : string, optional
            The file name to which the tree will be exported to as text.
            If the file doesn't exist already, it will be created. If the
            file already exists, it will be overwritten. If a file name
            is not specified, the tree will not be exported to a file.

        Returns
        -------
        self.exported_tree : string
             The exported tree stored as a string.
        """

        if type(language) == str:
            # using a preset language
            language_dict = self.__get_language_dict(language)
            
            if language_dict == {}:
                # inputted language is string, but string is not found in language presets
                raise ValueError('language preset \'' + language + '\' not found.')
            else:
                # inputted language is a string and is found in the language presets
                self.exported_tree = ''
                self.__writer(language_dict, feature_map, class_map, output_file_name)
                return self.exported_tree
        elif type(language) == dict:
            # using a custom language dictionary
            self.exported_tree = ''
            self.__writer(language, feature_map, class_map, output_file_name)
            return self.exported_tree
        else:
            # unknown input type (not string or dictionary)
            raise TypeError(str(language) + ' is an invalid language input.')
            
    def get_languages(self):
        """Get languages that have an available preset.
        ----------

        Returns
        -------
        presets : list of strings of shape (n_languages,)
             The languages that have presets available.
        """

        # reading the data from the file
        stream = pkg_resources.resource_filename('johnny_appleseed', 'data/language_dicts.dat')

        with open(stream) as f:
            data = f.read()

        # reconstructing the data as a dictionary of dictionaries
        language_dicts = json.loads(data)
        
        return [language_dict['name'] for language_dict in language_dicts['languages']]
    
    def get_language_preset(self, language):
        """Get the preset properties from a desired language.
        ----------
        language : string
            The language that the preset properties will be retrieved
            for.

        Returns
        -------
        language_dict : dictionary
             The dictionary containing the properties of the language
             preset.
        """

        if type(language) == str:
            # using a preset language
            language_dict = self.__get_language_dict(language)
            
            if language_dict == {}:
                # inputted language is string, but string is not found in language presets
                raise ValueError('language preset \'' + language + '\' not found.')
                
            return language_dict
        else:
            # unknown input type (not string or dictionary)
            raise TypeError(str(language) + ' is an invalid language input.')
        