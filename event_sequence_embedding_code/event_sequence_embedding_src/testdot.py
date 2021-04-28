# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:11:17 2019

@author: tsun04
"""

from subprocess import call




call(['dot', '-Tpng', "\C:\\Users\\tsun04\\event_sequence_embedding\\src\\forest.dot", '-o', 'forest.png', '-Gdpi=600'])



from subprocess import check_call
check_call(['dot','-Tpng','forest.dot','-o','OutputFile.png'])



from xgboost import XGBClassifiers


from sklearn.xgboost import XGBClassifiers