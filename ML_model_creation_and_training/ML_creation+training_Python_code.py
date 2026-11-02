import sqlite3
import numpy
import pandas
import tensorflow as TF
from sklearn.preprocessing import LabelEncoder

DB_URL = 'G:\\ML+SQL_project\\DB\\first_database.db'

db_connection = sqlite3.connect(DB_URL)
cursor_for_database = db_connection.cursor()

cursor_for_database.execute("SELECT * FROM housing_pricing_1;")
whole_test_dataset = cursor_for_database.fetchall()
cursor_for_database.execute("SELECT * FROM housing_pricing_2;")
whole_train_dataset = cursor_for_database.fetchall()
maybe_trainset = cursor_for_database.fetchone()

# cursor.description holds a Nx7 array where each row contain info about 1 column of query result
# information on 0-th index is column's name
# done because sqlite3 query results don't contain column names/titles
description_of_cursor = cursor_for_database.description
all_column_names_1 = [info[0] for info in description_of_cursor]

cursor_for_database.close()
db_connection.close()

# training set has the column with sale price, test set doesn't have it
all_column_names_2 = all_column_names_1[:80]

trainset_DF = pandas.DataFrame(data=whole_train_dataset,columns=all_column_names_1)
testset_DF = pandas.DataFrame(data=whole_test_dataset,columns=all_column_names_2)
encod = LabelEncoder()

# 43 out of 81 columns aren't purely numerical
nonnumerical_columns_names = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
                              'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
                              'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                              'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',
                              'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType',
                              'SaleCondition']

copied_train_DF = trainset_DF

encoding_legend = pandas.DataFrame(data=[0,0,0],columns=["first"])
for col in nonnumerical_columns_names:
    encod.fit(trainset_DF[col])
    #trainset_DF[col] = encod.transform(trainset_DF[col])
    column_legend_DF = pandas.DataFrame(data=encod.classes_,columns=[str(col)])
    if col == 'MSZoning':
        encoding_legend = column_legend_DF
    else:
        encoding_legend = pandas.concat(objs=[encoding_legend,column_legend_DF],axis=1,ignore_index=False)

for col in nonnumerical_columns_names:
    trainset_DF[col] = encod.fit_transform(trainset_DF[col])

"""
columns to get rid of (name - index):
 - Id - 0
 - LotFrontage - 3
 - Neighborhood - 12
 - MasVnrArea - 26
 - BsmtExposure - 32
 - BsmtUnfSF - 37
 - GarageYrBlt - 59
 - MiscVal - 75
"""
"""
truncated_trainset = numpy.delete(
    arr=whole_train_dataset,
    obj=[0,3,12,26,32,37,59,75],
    axis=1)
truncated_testset = numpy.delete(
    arr=whole_test_dataset,
    obj=[0,3,12,26,32,37,59,75],
    axis=1)
"""
truncated_trainset = trainset_DF.drop(
    labels=['Id','LotFrontage','Neighborhood','MasVnrArea','BsmtExposure','BsmtUnfSF','GarageYrBlt','MiscVal'],
    axis=1)

sets_only_for_training = truncated_trainset.sample(frac=0.75)
sets_only_for_validation = truncated_trainset.drop(sets_only_for_training.index)

traintrain_data = sets_only_for_training.drop(labels='SalePrice',axis=1)
traintrain_prices = sets_only_for_training['SalePrice']

trainval_data = sets_only_for_validation.drop(labels='SalePrice',axis=1)
trainval_prices = sets_only_for_validation['SalePrice']

# "model"
shape_of_input = [traintrain_data.shape[1]] # 1-cell array
neural_network = TF.keras.Sequential([
    TF.keras.layers.Dense(units=32, activation='relu', input_shape=shape_of_input), # 81-8=73
    TF.keras.layers.Dense(units=32, activation='relu'),
    TF.keras.layers.Dense(units=1),
])

neural_network.compile(
    optimizer='adam',
    loss='mae'
    )

# trying the below to fix the "ValueError: Invalid dtype:str" error when running "model.fit()" function
traintrain_data = traintrain_data.astype(float)
traintrain_prices = traintrain_prices.astype(float)
trainval_data = trainval_data.astype(float)
trainval_prices = trainval_prices.astype(float)

TT_d = numpy.array(traintrain_data)
TT_p = numpy.array(traintrain_prices)
TV_d = numpy.array(trainval_data)
TV_p = numpy.array(trainval_prices)

losses_from_training = neural_network.fit(
    TT_d,
    TT_p,
    validation_data=(TV_d, TV_p),
    #batch_size=256,
    epochs=40
)

print("----\n----")
print(type(losses_from_training))
losses_DF = pandas.DataFrame(losses_from_training.history)
losses_DF.loc[:,['loss','val_loss']].plot()

# www.leoniemonigatti.com/blog/save-and-load-neural-networks.html
neural_network.save("save_aparts_1.keras")
encoding_legend.to_csv(path_or_buf="encoded_labels_in_order.csv",sep=';',index=False)