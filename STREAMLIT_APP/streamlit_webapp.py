import numpy as np
import streamlit as st
import pandas
import keras
from pathlib import Path

#TODO: check why there's no "MSSubClass" and "OverallQual" in array with encoding
# ---------------------------------------------------------
# FUNCTION = Load model
# ---------------------------------------------------------

@st.cache_resource
def load_model(path_to_parent:str):
    full_model_path = Path(path_to_parent) / "model" / "save_aparts_1.keras"
    #return keras.saving.load_model(MODEL_PATH)
    #model = keras.saving.load_model("G:\\ML+SQL_project\\save\\save_aparts_1.keras")
    loaded_model = keras.saving.load_model(filepath=full_model_path)
    return loaded_model

# ---------------------------------------------------------
# Page
# ---------------------------------------------------------

string_path_to_parent_directory = str(Path(__file__).parent)

model = load_model(string_path_to_parent_directory)

st.set_page_config(
    page_title="ML Prediction App",
    page_icon="ðŸ¤–",  #should I change this or not?
    layout="centered"
)

st.title("Machine Learning Prediction")
st.write("Set the values below and click Calculate.")
st.write("ML model will make prediction based on values given by you")

# ---------------------------------------------------------
# Selectors for user's use
# ---------------------------------------------------------

advanced_arguments = st.checkbox("Advanced arguments")
left_column, right_column = st.columns(2)

#  based on this:  evidence.dev/learn/streamlit-selectbox#using-selectbox-with-dataframes

string_c_1="""
MSSubClass_DF = pandas.DataFrame({
    'symbol': ['20', '30', '40', '45', '50', '60', '70', '75', '80', '85', '90', '120', '150', '160', '180', '190'],
    'description': ['1-STORY 1946 & NEWER ALL STYLES', '1-STORY 1945 & OLDER', '1-STORY W/FINISHED ATTIC ALL AGES', 
                    '1-1/2 STORY - UNFINISHED ALL AGES', '1-1/2 STORY FINISHED ALL AGES', '2-STORY 1946 & NEWER', '2-STORY 1945 & OLDER', 
                    '2-1/2 STORY ALL AGES', 'SPLIT OR MULTI-LEVEL', 'SPLIT FOYER', 'DUPLEX - ALL STYLES AND AGES', 
                    '1-STORY PUD (Planned Unit Development) - 1946 & NEWER', '1-1/2 STORY PUD - ALL AGES', '2-STORY PUD - 1946 & NEWER', 
                    'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER', '2 FAMILY CONVERSION - ALL STYLES AND AGES']
})
"""


MSZoning_DF = pandas.DataFrame({
    'symbol':["A","C (all)","FV","I","RH","RL","RP","RM"],
    'description':["Agriculture","Commercial","Floating Village Residential","Industrial",
                   "Residential High Density","Residential Low Density","Residential Low Density Park",
                   "Residential Medium Density"]
})

string_c_11="""
MSZoning_DF = pandas.DataFrame({
    'symbol':["C","FV","RH","RL","RM"],
    'description':["Commercial","Floating Village Residential",
                   "Residential High Density","Residential Low Density",
                   "Residential Medium Density"]
})
"""

Street_DF = pandas.DataFrame({
    'symbol': ["Grvl","Pave"],
    'description': ["Gravel","Paved"]
})

Alley_DF = pandas.DataFrame({
    'symbol': ["Grvl","Pave","NA"],
    'description': ["Gravel","Paved","No alley access"]
})

LotShape_DF = pandas.DataFrame({
    'symbol': ['Reg', 'IR1', 'IR2', 'IR3'],
    'description': ['Regular', 'Slightly irregular', 'Moderately Irregular', 'Irregular']
})

LandContour_DF = pandas.DataFrame({
    'symbol': ['Lvl', 'Bnk', 'HLS', 'Low'],
    'description': ['Near Flat/Level', 'Banked - Quick and significant rise from street grade to building', 'Hillside - Significant slope from side to side', 'Depression']
})

Utilities_DF = pandas.DataFrame({
    'symbol': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
    'description': ['All public Utilities (E,G,W,& S)', 'Electricity, Gas, and Water (Septic Tank)', 'Electricity and Gas Only', 'Electricity only']
})

LotConfig_DF = pandas.DataFrame({
    'symbol': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
    'description': ['Inside lot', 'Corner lot', 'Cul-de-sac', 'Frontage on 2 sides of property', 'Frontage on 3 sides of property']
})

LandSlope_DF = pandas.DataFrame({
    'symbol': ['0.0', '1.0', '2.0'],
    'description': ['Gentle slope', 'Moderate Slope', 'Severe Slope']
})

Neighborhood_DF = pandas.DataFrame({
    'symbol': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'],
    'description': ['Bloomington Heights', 'Bluestem', 'Briardale', 'Brookside', 'Clear Creek', 'College Creek', 'Crawford', 'Edwards', 'Gilbert', 'Iowa DOT and Rail Road', 'Meadow Village', 'Mitchell', 'North Ames', 'Northridge', 'Northpark Villa', 'Northridge Heights', 'Northwest Ames', 'Old Town', 'South & West of Iowa State University', 'Sawyer', 'Sawyer West', 'Somerset', 'Stone Brook', 'Timberland', 'Veenker']
})

Condition1_DF = pandas.DataFrame({
    'symbol': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
    'description': ['Adjacent to arterial street', 'Adjacent to feeder street', 'Normal', "Within 200' of North-South Railroad", 'Adjacent to North-South Railroad', 'Near positive off-site feature--park, greenbelt, etc.', 'Adjacent to postive off-site feature', "Within 200' of East-West Railroad", 'Adjacent to East-West Railroad']
})

Condition2_DF = pandas.DataFrame({
    'symbol': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
    'description': ['Adjacent to arterial street', 'Adjacent to feeder street', 'Normal', "Within 200' of North-South Railroad", 'Adjacent to North-South Railroad', 'Near positive off-site feature--park, greenbelt, etc.', 'Adjacent to postive off-site feature', "Within 200' of East-West Railroad", 'Adjacent to East-West Railroad']
})

BldgType_DF = pandas.DataFrame({
    'symbol': ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'],
    'description': ['Single-family Detached', 'Two-family Conversion; originally built as one-family dwelling', 'Duplex', 'Townhouse End Unit', 'Townhouse Inside Unit']
})

HouseStyle_DF = pandas.DataFrame({
    'symbol': ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
    'description': ['One story', 'One and one-half story: 2nd level finished', 'One and one-half story: 2nd level unfinished', 'Two story', 'Two and one-half story: 2nd level finished', 'Two and one-half story: 2nd level unfinished', 'Split Foyer', 'Split Level']
})

string_c_2="""
OverallQual_DF = pandas.DataFrame({
    'symbol': ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1'],
    'description': ['Very Excellent', 'Excellent', 'Very Good', 'Good', 'Above Average', 'Average', 'Below Average', 'Fair', 'Poor', 'Very Poor']
})
"""

string_c_3="""
OverallCond_DF = pandas.DataFrame({
    'symbol': ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1'],
    'description': ['Very Excellent', 'Excellent', 'Very Good', 'Good', 'Above Average', 'Average', 'Below Average', 'Fair', 'Poor', 'Very Poor']
})
"""

RoofStyle_DF = pandas.DataFrame({
    'symbol': ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
    'description': ['Flat', 'Gable', 'Gabrel (Barn)', 'Hip', 'Mansard', 'Shed']
})

RoofMatl_DF = pandas.DataFrame({
    'symbol': ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
    'description': ['Clay or Tile', 'Standard (Composite) Shingle', 'Membrane', 'Metal', 'Roll', 'Gravel & Tar', 'Wood Shakes', 'Wood Shingles']
})

Exterior1st_DF = pandas.DataFrame({
    'symbol': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing'],
    'description': ['Asbestos Shingles', 'Asphalt Shingles', 'Brick Common', 'Brick Face', 'Cinder Block', 'Cement Board', 'Hard Board', 'Imitation Stucco', 'Metal Siding', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'Vinyl Siding', 'Sdng\tWood Siding', 'Wood Shingles']
})

Exterior2nd_DF = pandas.DataFrame({
    'symbol': ['AsbShng', 'AsphShn', 'Brk Cmn', 'BrkFace', 'CBlock', 'CmentBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'Wd Shng'],
    'description': ['Asbestos Shingles', 'Asphalt Shingles', 'Brick Common', 'Brick Face', 'Cinder Block', 'Cement Board', 'Hard Board', 'Imitation Stucco', 'Metal Siding', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'Vinyl Siding', 'Sdng\tWood Siding', 'Wood Shingles']
})

MasVnrType_DF = pandas.DataFrame({
    'symbol': ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'],
    'description': ['Brick Common', 'Brick Face', 'Cinder Block', 'None', 'Stone']
})

ExterQual_DF = pandas.DataFrame({
    'symbol': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'description': ['Excellent', 'Good', 'Average/Typical', 'Fair', 'Poor']
})

ExterCond_DF = pandas.DataFrame({
    'symbol': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'description': ['Excellent', 'Good', 'Average/Typical', 'Fair', 'Poor']
})

Foundation_DF = pandas.DataFrame({
    'symbol': ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
    'description': ['Brick & Tile', 'Cinder Block', 'Poured Contrete', 'Slab', 'Stone', 'Wood']
})

BsmtQual_DF = pandas.DataFrame({
    'symbol': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'description': ['Excellent (100+ inches)', 'Good (90-99 inches)', 'Typical (80-89 inches)', 'Fair (70-79 inches)', 'Poor (<70 inches', 'No Basement']
})

BsmtCond_DF = pandas.DataFrame({
    'symbol': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'description': ['Excellent', 'Good', 'Typical - slight dampness allowed', 'Fair - dampness or some cracking or settling', 'Poor - Severe cracking, settling, or wetness', 'No Basement']
})

BsmtExposure_DF = pandas.DataFrame({
    'symbol': ['Gd', 'Av', 'Mn', 'No', 'NA'],
    'description': ['Good Exposure', 'Average Exposure (split levels or foyers typically score average or above)', 'Mimimum Exposure', 'No Exposure', 'No Basement']
})

BsmtFinType1_DF = pandas.DataFrame({
    'symbol': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'description': ['Good Living Quarters', 'Average Living Quarters', 'Below Average Living Quarters', 'Average Rec Room', 'Low Quality', 'Unfinshed', 'No Basement']
})

BsmtFinType2_DF = pandas.DataFrame({
    'symbol': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'description': ['Good Living Quarters', 'Average Living Quarters', 'Below Average Living Quarters', 'Average Rec Room', 'Low Quality', 'Unfinshed', 'No Basement']
})

Heating_DF = pandas.DataFrame({
    'symbol': ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
    'description': ['Floor Furnace', 'Gas forced warm air furnace', 'Gas hot water or steam heat', 'Gravity furnace', 'Hot water or steam heat other than gas', 'Wall furnace']
})

HeatingQC_DF = pandas.DataFrame({
    'symbol': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'description': ['Excellent', 'Good', 'Average/Typical', 'Fair', 'Poor']
})

CentralAir_DF = pandas.DataFrame({
    'symbol': ['N', 'Y'],
    'description': ['No', 'Yes']
})

Electrical_DF = pandas.DataFrame({
    'symbol': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
    'description': ['Standard Circuit Breakers & Romex', 'Fuse Box over 60 AMP and all Romex wiring (Average)', '60 AMP Fuse Box and mostly Romex wiring (Fair)', '60 AMP Fuse Box and mostly knob & tube wiring (poor)', 'Mixed']
})

KitchenQual_DF = pandas.DataFrame({
    'symbol': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'description': ['Excellent', 'Good', 'Typical/Average', 'Fair', 'Poor']
})

Functional_DF = pandas.DataFrame({
    'symbol': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
    'description': ['Typical Functionality', 'Minor Deductions 1', 'Minor Deductions 2', 'Moderate Deductions', 'Major Deductions 1', 'Major Deductions 2', 'Severely Damaged', 'Salvage only']
})

FireplaceQu_DF = pandas.DataFrame({
    'symbol': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'description': ['Excellent - Exceptional Masonry Fireplace', 'Good - Masonry Fireplace in main level', 'Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement', 'Fair - Prefabricated Fireplace in basement', 'Poor - Ben Franklin Stove', 'No Fireplace']
})

GarageType_DF = pandas.DataFrame({
    'symbol': ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'],
    'description': ['More than one type of garage', 'Attached to home', 'Basement Garage', 'Built-In (Garage part of house - typically has room above garage)', 'Car Port', 'Detached from home', 'No Garage']
})

GarageFinish_DF = pandas.DataFrame({
    'symbol': ['Fin', 'RFn', 'Unf', 'NA'],
    'description': ['Finished', 'Rough Finished', 'Unfinished', 'No Garage']
})

GarageQual_DF = pandas.DataFrame({
    'symbol': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'description': ['Excellent', 'Good', 'Typical/Average', 'Fair', 'Poor', 'No Garage']
})

GarageCond_DF = pandas.DataFrame({
    'symbol': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'description': ['Excellent', 'Good', 'Typical/Average', 'Fair', 'Poor', 'No Garage']
})

PavedDrive_DF = pandas.DataFrame({
    'symbol': ['Y', 'P', 'N'],
    'description': ['Paved', 'Partial Pavement', 'Dirt/Gravel']
})

PoolQC_DF = pandas.DataFrame({
    'symbol': ['Ex', 'Gd', 'TA', 'Fa', 'NA'],
    'description': ['Excellent', 'Good', 'Average/Typical', 'Fair', 'No Pool']
})

Fence_DF = pandas.DataFrame({
    'symbol': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA'],
    'description': ['Good Privacy', 'Minimum Privacy', 'Good Wood', 'Minimum Wood/Wire', 'No Fence']
})

MiscFeature_DF = pandas.DataFrame({
    'symbol': ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA'],
    'description': ['Elevator', '2nd Garage (if not described in garage section)', 'Other', 'Shed (over 100 SF)', 'Tennis Court', 'None']
})

SaleType_DF = pandas.DataFrame({
    'symbol': ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'],
    'description': ['Warranty Deed - Conventional', 'Warranty Deed - Cash', 'Warranty Deed - VA Loan', 'Home just constructed and sold', 'Court Officer Deed/Estate', 'Contract 15% Down payment regular terms', 'Contract Low Down payment and low interest', 'Contract Low Interest', 'Contract Low Down', 'Other']
})

SaleCondition_DF = pandas.DataFrame({
    'symbol': ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'],
    'description': ['Normal Sale', 'Abnormal Sale -  trade, foreclosure, short sale', 'Adjoining Land Purchase', 'Allocation - two linked properties with separate deeds, typically condo with a garage unit', 'Sale between family members', 'Home was not completed when last assessed (associated with New Homes)']
})


#-----------------------------------------------
#-----------------------------------------------

s_LotFrontage = 60
s_Neighborhood = Neighborhood_DF["description"].iloc[0]
s_MasVnrArea = 0
s_BsmtExposure = BsmtExposure_DF["description"].iloc[0]
s_BsmtUnfSF = 0
s_GarageYrBlt = 1980
s_MiscVal = 0

if advanced_arguments:
    # Controls ordered as in data_description.txt
    
    with left_column:
        s_MSSubClass = st.slider(
            label="Type of dwelling",
            min_value=20,
            max_value=190,
            value=60
        )
    
    with right_column:
        s_MSZoning = st.selectbox(
            label='Identifies the general zoning classification of the sale.',
            options=MSZoning_DF['description'],
            placeholder="select one..."
        )
    
    string_c_lot_frontage = """
    with left_column:
        s_LotFrontage = st.slider(
            label='Linear feet of street connected to property',
            min_value=0,
            max_value=300,
            value=60
        )
    """
    
    with right_column:
        s_LotArea = st.slider(
            label='Lot size in square feet',
            min_value=0,
            max_value=220000,
            value=10000,
            step=100
        )
    
    with left_column:
        s_Street = st.selectbox(
            label='Type of road access to property',
            options=Street_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_Alley = st.selectbox(
            label='Type of alley access to property',
            options=Alley_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_LotShape = st.selectbox(
            label='General shape of property',
            options=LotShape_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_LandContour = st.selectbox(
            label='Flatness of the property',
            options=LandContour_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_Utilities = st.selectbox(
            label='Type of utilities available',
            options=Utilities_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_LotConfig = st.selectbox(
            label='Lot configuration',
            options=LotConfig_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_LandSlope = st.selectbox(
            label='Slope of property',
            options=LandSlope_DF['description'],
            placeholder="select one..."
        )
    
    string_c_neighborhood = """
    with right_column:
        s_Neighborhood = st.selectbox(
            label='Physical locations within Ames city limits',
            options=Neighborhood_DF['description'],
            placeholder="select one..."
        )
    """
    
    with left_column:
        s_Condition1 = st.selectbox(
            label='Proximity to various conditions',
            options=Condition1_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_Condition2 = st.selectbox(
            label='Proximity to various conditions (if more than one is present)',
            options=Condition2_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_BldgType = st.selectbox(
            label='Type of dwelling',
            options=BldgType_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_HouseStyle = st.selectbox(
            label='Style of dwelling',
            options=HouseStyle_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_OverallQual = st.slider(
            label="Overall material and finish quality",
            min_value=1,
            max_value=10,
            value=5
        )
    
    with right_column:
        s_OverallCond = st.slider(
            label="Overall condition",
            min_value=1,
            max_value=10,
            value=5
        )
    
    with left_column:
        s_YearBuilt = st.slider(
            label='Original construction date',
            min_value=1872,
            max_value=2010,
            value=1970
        )
    
    with right_column:
        s_YearRemodAdd = st.slider(
            label='Remodel date',
            min_value=1950,
            max_value=2010,
            value=1970
        )
    
    with left_column:
        s_RoofStyle = st.selectbox(
            label='Type of roof',
            options=RoofStyle_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_RoofMatl = st.selectbox(
            label='Roof material',
            options=RoofMatl_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_Exterior1st = st.selectbox(
            label='Exterior covering on house',
            options=Exterior1st_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_Exterior2nd = st.selectbox(
            label='Exterior covering on house (if more than one material)',
            options=Exterior2nd_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_MasVnrType = st.selectbox(
            label='Masonry veneer type',
            options=MasVnrType_DF['description'],
            placeholder="select one..."
        )
    
    string_c_mas_vnr_area = """
    with left_column:
        s_MasVnrArea = st.slider(
            label='Masonry veneer area in square feet',
            min_value=0,
            max_value=1600,
            value=0,
            step=10
        )
    """
    
    with right_column:
        s_ExterQual = st.selectbox(
            label='Evaluates the quality of the material on the exterior',
            options=ExterQual_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_ExterCond = st.selectbox(
            label='Evaluates the present condition of the material on the exterior',
            options=ExterCond_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_Foundation = st.selectbox(
            label='Type of foundation',
            options=Foundation_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_BsmtQual = st.selectbox(
            label='Evaluates the height of the basement',
            options=BsmtQual_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_BsmtCond = st.selectbox(
            label='Evaluates the general condition of the basement',
            options=BsmtCond_DF['description'],
            placeholder="select one..."
        )
    
    string_c_bsmt_exposure = """
    with left_column:
        s_BsmtExposure = st.selectbox(
            label='Refers to walkout or garden level walls',
            options=BsmtExposure_DF['description'],
            placeholder="select one..."
        )
    """
    
    with right_column:
        s_BsmtFinType1 = st.selectbox(
            label='Rating of basement finished area',
            options=BsmtFinType1_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_BsmtFinSF1 = st.slider(
            label='Type 1 finished square feet',
            min_value=0,
            max_value=6000,
            value=0,
            step=10
        )
    
    with left_column:
        s_BsmtFinType2 = st.selectbox(
            label='Rating of basement finished area (if multiple types)',
            options=BsmtFinType2_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_BsmtFinSF2 = st.slider(
            label='Type 2 finished square feet',
            min_value=0,
            max_value=1600,
            value=0,
            step=10
        )
    
    string_c_bsmt_unf_sf = """
    with right_column:
        s_BsmtUnfSF = st.slider(
            label='Unfinished square feet of basement area',
            min_value=0,
            max_value=2500,
            value=0,
            step=10
        )
    """
    
    with left_column:
        s_TotalBsmtSF = st.slider(
            label='Total square feet of basement area',
            min_value=0,
            max_value=6500,
            value=1000,
            step=10
        )
    
    with right_column:
        s_Heating = st.selectbox(
            label='Type of heating',
            options=Heating_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_HeatingQC = st.selectbox(
            label='Heating quality and condition',
            options=HeatingQC_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_CentralAir = st.selectbox(
            label='Central air conditioning',
            options=CentralAir_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_Electrical = st.selectbox(
            label='Electrical system',
            options=Electrical_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_1stFlrSF = st.slider(
            label='First Floor square feet',
            min_value=0,
            max_value=4000,
            value=1000,
            step=10
        )
    
    with left_column:
        s_2ndFlrSF = st.slider(
            label='Second floor square feet',
            min_value=0,
            max_value=2100,
            value=0,
            step=10
        )
    
    with right_column:
        s_LowQualFinSF = st.slider(
            label='Low quality finished square feet (all floors)',
            min_value=0,
            max_value=600,
            value=0,
            step=10
        )
    
    with left_column:
        s_GrLivArea = st.slider(
            label='Above grade (ground) living area square feet',
            min_value=0,
            max_value=6000,
            value=1500,
            step=10
        )
    
    with right_column:
        s_BsmtFullBath = st.slider(
            label='Basement full bathrooms',
            min_value=0,
            max_value=3,
            value=0
        )
    
    with left_column:
        s_BsmtHalfBath = st.slider(
            label='Basement half bathrooms',
            min_value=0,
            max_value=2,
            value=0
        )
    
    with right_column:
        s_FullBath = st.slider(
            label='Full bathrooms above grade',
            min_value=0,
            max_value=4,
            value=1
        )
    
    with left_column:
        s_HalfBath = st.slider(
            label='Half baths above grade',
            min_value=0,
            max_value=2,
            value=0
        )
    
    with right_column:
        s_BedroomAbvGr = st.slider(
            label='Bedrooms above grade',
            min_value=0,
            max_value=8,
            value=3
        )
    
    with left_column:
        s_KitchenAbvGr = st.slider(
            label='Kitchens above grade',
            min_value=0,
            max_value=3,
            value=1
        )
    
    with right_column:
        s_KitchenQual = st.selectbox(
            label='Kitchen quality',
            options=KitchenQual_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_TotRmsAbvGrd = st.slider(
            label='Total rooms above grade',
            min_value=0,
            max_value=15,
            value=6
        )
    
    with left_column:
        s_Functional = st.selectbox(
            label='Home functionality (Assume typical unless deductions are warranted)',
            options=Functional_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_Fireplaces = st.slider(
            label='Number of fireplaces',
            min_value=0,
            max_value=4,
            value=0
        )
    
    with right_column:
        s_FireplaceQu = st.selectbox(
            label='Fireplace quality',
            options=FireplaceQu_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_GarageType = st.selectbox(
            label='Garage location',
            options=GarageType_DF['description'],
            placeholder="select one..."
        )
    
    string_c_garage_yr_blt = """
    with right_column:
        s_GarageYrBlt = st.slider(
            label='Year garage was built',
            min_value=1900,
            max_value=2010,
            value=1980
        )
    """
    
    with right_column:
        s_GarageFinish = st.selectbox(
            label='Interior finish of the garage',
            options=GarageFinish_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_GarageCars = st.slider(
            label='Size of garage in car capacity',
            min_value=0,
            max_value=5,
            value=2
        )
    
    with right_column:
        s_GarageArea = st.slider(
            label='Size of garage in square feet',
            min_value=0,
            max_value=1500,
            value=400,
            step=10
        )
    
    with left_column:
        s_GarageQual = st.selectbox(
            label='Garage quality',
            options=GarageQual_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_GarageCond = st.selectbox(
            label='Garage condition',
            options=GarageCond_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_PavedDrive = st.selectbox(
            label='Paved driveway',
            options=PavedDrive_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_WoodDeckSF = st.slider(
            label='Wood deck area in square feet',
            min_value=0,
            max_value=900,
            value=0,
            step=10
        )
    
    with right_column:
        s_OpenPorchSF = st.slider(
            label='Open porch area in square feet',
            min_value=0,
            max_value=600,
            value=0,
            step=10
        )
    
    with left_column:
        s_EnclosedPorch = st.slider(
            label='Enclosed porch area in square feet',
            min_value=0,
            max_value=600,
            value=0,
            step=10
        )
    
    with right_column:
        s_3SsnPorch = st.slider(
            label='Three season porch area in square feet',
            min_value=0,
            max_value=500,
            value=0,
            step=10
        )
    
    with left_column:
        s_ScreenPorch = st.slider(
            label='Screen porch area in square feet',
            min_value=0,
            max_value=550,
            value=0,
            step=10
        )
    
    with right_column:
        s_PoolArea = st.slider(
            label='Pool area in square feet',
            min_value=0,
            max_value=800,
            value=0,
            step=10
        )
    
    with right_column:
        s_PoolQC = st.selectbox(
            label='Pool quality',
            options=PoolQC_DF['description'],
            placeholder="select one..."
        )
    
    with left_column:
        s_Fence = st.selectbox(
            label='Fence quality',
            options=Fence_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_MiscFeature = st.selectbox(
            label='Miscellaneous feature not covered in other categories',
            options=MiscFeature_DF['description'],
            placeholder="select one..."
        )
    
    string_c_misc_val = """
    with left_column:
        s_MiscVal = st.slider(
            label='Value of miscellaneous feature',
            min_value=0,
            max_value=16000,
            value=0,
            step=100
        )
    """
    
    with right_column:
        s_MoSold = st.slider(
            label='Month Sold (MM)',
            min_value=1,
            max_value=12,
            value=6
        )
    
    with left_column:
        s_YrSold = st.slider(
            label='Year Sold (YYYY)',
            min_value=2006,
            max_value=2010,
            value=2008
        )
    
    with left_column:
        s_SaleType = st.selectbox(
            label='Type of sale',
            options=SaleType_DF['description'],
            placeholder="select one..."
        )
    
    with right_column:
        s_SaleCondition = st.selectbox(
            label='Condition of sale',
            options=SaleCondition_DF['description'],
            placeholder="select one..."
        )
    
    # ---------------------------------------------------------
else:
    basic_left, basic_right = st.columns(2)
    s_MSSubClass = 60
    s_OverallQual = 5
    s_OverallCond = 5

    with basic_left:
        s_MSZoning = st.selectbox(
            label="Identifies the general zoning classification of the sale.",
            options=MSZoning_DF["description"],
            key="basic_MSZoning"
        )
    with basic_right:
        s_Street = st.selectbox(
            label="Type of road access to property",
            options=Street_DF["description"],
            key="basic_Street"
        )
    with basic_left:
        s_BldgType = st.selectbox(
            label="Type of dwelling",
            options=BldgType_DF["description"],
            key="basic_BldgType"
        )
    with basic_right:
        s_HouseStyle = st.selectbox(
            label="Style of dwelling",
            options=HouseStyle_DF["description"],
            key="basic_HouseStyle"
        )
    with basic_left:
        s_BsmtFinType1 = st.selectbox(
            label="Rating of basement finished area",
            options=BsmtFinType1_DF["description"],
            key="basic_BsmtFinType1"
        )
    with basic_right:
        s_Heating = st.selectbox(
            label="Type of heating",
            options=Heating_DF["description"],
            key="basic_Heating"
        )
    with basic_left:
        s_TotalBsmtSF = st.slider(
            label="Total square feet of basement area",
            min_value=0,
            max_value=6500,
            value=1000,
            step=10,
            key="basic_TotalBsmtSF"
        )
    with basic_right:
        s_GarageType = st.selectbox(
            label="Garage location",
            options=GarageType_DF["description"],
            key="basic_GarageType"
        )
    with basic_left:
        s_GarageCars = st.slider(
            label="Size of garage in car capacity",
            min_value=0,
            max_value=5,
            value=2,
            key="basic_GarageCars"
        )
    with basic_right:
        s_LotArea = st.slider(
            label="Lot size in square feet",
            min_value=0,
            max_value=220000,
            value=10000,
            step=100,
            key="basic_LotArea"
        )
    with basic_left:
        s_GrLivArea = st.slider(
            label="Above grade (ground) living area square feet",
            min_value=0,
            max_value=6000,
            value=1500,
            step=10,
            key="basic_GrLivArea"
        )
    with basic_right:
        s_FullBath = st.slider(
            label="Full bathrooms above grade",
            min_value=0,
            max_value=4,
            value=1,
            key="basic_FullBath"
        )
    with basic_left:
        s_TotRmsAbvGrd = st.slider(
            label="Total rooms above grade",
            min_value=0,
            max_value=15,
            value=6,
            key="basic_TotRmsAbvGrd"
        )

    s_LotFrontage = 60
    s_Alley = Alley_DF["description"].iloc[0]
    s_LotShape = LotShape_DF["description"].iloc[0]
    s_LandContour = LandContour_DF["description"].iloc[0]
    s_Utilities = Utilities_DF["description"].iloc[0]
    s_LotConfig = LotConfig_DF["description"].iloc[0]
    s_LandSlope = LandSlope_DF["description"].iloc[0]
    s_Neighborhood = Neighborhood_DF["description"].iloc[0]
    s_Condition1 = Condition1_DF["description"].iloc[0]
    s_Condition2 = Condition2_DF["description"].iloc[0]
    s_YearBuilt = 1970
    s_YearRemodAdd = 1970
    s_RoofStyle = RoofStyle_DF["description"].iloc[0]
    s_RoofMatl = RoofMatl_DF["description"].iloc[0]
    s_Exterior1st = Exterior1st_DF["description"].iloc[0]
    s_Exterior2nd = Exterior2nd_DF["description"].iloc[0]
    s_MasVnrType = MasVnrType_DF["description"].iloc[0]
    s_MasVnrArea = 0
    s_ExterQual = ExterQual_DF["description"].iloc[0]
    s_ExterCond = ExterCond_DF["description"].iloc[0]
    s_Foundation = Foundation_DF["description"].iloc[0]
    s_BsmtQual = BsmtQual_DF["description"].iloc[0]
    s_BsmtCond = BsmtCond_DF["description"].iloc[0]
    s_BsmtExposure = BsmtExposure_DF["description"].iloc[0]
    s_BsmtFinSF1 = 0
    s_BsmtFinType2 = BsmtFinType2_DF["description"].iloc[0]
    s_BsmtFinSF2 = 0
    s_BsmtUnfSF = 0
    s_HeatingQC = HeatingQC_DF["description"].iloc[0]
    s_CentralAir = CentralAir_DF["description"].iloc[0]
    s_Electrical = Electrical_DF["description"].iloc[0]
    s_1stFlrSF = 1000
    s_2ndFlrSF = 0
    s_LowQualFinSF = 0
    s_BsmtFullBath = 0
    s_BsmtHalfBath = 0
    s_HalfBath = 0
    s_BedroomAbvGr = 3
    s_KitchenAbvGr = 1
    s_KitchenQual = KitchenQual_DF["description"].iloc[0]
    s_Functional = Functional_DF["description"].iloc[0]
    s_Fireplaces = 0
    s_FireplaceQu = FireplaceQu_DF["description"].iloc[0]
    s_GarageYrBlt = 1980
    s_GarageFinish = GarageFinish_DF["description"].iloc[0]
    s_GarageArea = 400
    s_GarageQual = GarageQual_DF["description"].iloc[0]
    s_GarageCond = GarageCond_DF["description"].iloc[0]
    s_PavedDrive = PavedDrive_DF["description"].iloc[0]
    s_WoodDeckSF = 0
    s_OpenPorchSF = 0
    s_EnclosedPorch = 0
    s_3SsnPorch = 0
    s_ScreenPorch = 0
    s_PoolArea = 0
    s_PoolQC = PoolQC_DF["description"].iloc[0]
    s_Fence = Fence_DF["description"].iloc[0]
    s_MiscFeature = MiscFeature_DF["description"].iloc[0]
    s_MiscVal = 0
    s_MoSold = 6
    s_YrSold = 2008
    s_SaleType = SaleType_DF["description"].iloc[0]
    s_SaleCondition = SaleCondition_DF["description"].iloc[0]

# saving the choices
# ---------------------------------------------------------

f_MSZoning = MSZoning_DF[MSZoning_DF['description'] == s_MSZoning].iat[0,0]
f_Street = Street_DF[Street_DF['description'] == s_Street].iat[0,0]
f_Alley = Alley_DF[Alley_DF['description'] == s_Alley].iat[0,0]
f_LotShape = LotShape_DF[LotShape_DF['description'] == s_LotShape].iat[0,0]
f_LandContour = LandContour_DF[LandContour_DF['description'] == s_LandContour].iat[0,0]
f_Utilities = Utilities_DF[Utilities_DF['description'] == s_Utilities].iat[0,0]
f_LotConfig = LotConfig_DF[LotConfig_DF['description'] == s_LotConfig].iat[0,0]
f_LandSlope = LandSlope_DF[LandSlope_DF['description'] == s_LandSlope].iat[0,0]
f_Condition1 = Condition1_DF[Condition1_DF['description'] == s_Condition1].iat[0,0]
f_Condition2 = Condition2_DF[Condition2_DF['description'] == s_Condition2].iat[0,0]
f_BldgType = BldgType_DF[BldgType_DF['description'] == s_BldgType].iat[0,0]
f_HouseStyle = HouseStyle_DF[HouseStyle_DF['description'] == s_HouseStyle].iat[0,0]
f_RoofStyle = RoofStyle_DF[RoofStyle_DF['description'] == s_RoofStyle].iat[0,0]
f_RoofMatl = RoofMatl_DF[RoofMatl_DF['description'] == s_RoofMatl].iat[0,0]
f_Exterior1st = Exterior1st_DF[Exterior1st_DF['description'] == s_Exterior1st].iat[0,0]
f_Exterior2nd = Exterior2nd_DF[Exterior2nd_DF['description'] == s_Exterior2nd].iat[0,0]
f_MasVnrType = MasVnrType_DF[MasVnrType_DF['description'] == s_MasVnrType].iat[0,0]
f_ExterQual = ExterQual_DF[ExterQual_DF['description'] == s_ExterQual].iat[0,0]
f_ExterCond = ExterCond_DF[ExterCond_DF['description'] == s_ExterCond].iat[0,0]
f_Foundation = Foundation_DF[Foundation_DF['description'] == s_Foundation].iat[0,0]
f_BsmtQual = BsmtQual_DF[BsmtQual_DF['description'] == s_BsmtQual].iat[0,0]
f_BsmtCond = BsmtCond_DF[BsmtCond_DF['description'] == s_BsmtCond].iat[0,0]
f_BsmtFinType1 = BsmtFinType1_DF[BsmtFinType1_DF['description'] == s_BsmtFinType1].iat[0,0]
f_BsmtFinType2 = BsmtFinType2_DF[BsmtFinType2_DF['description'] == s_BsmtFinType2].iat[0,0]
f_Heating = Heating_DF[Heating_DF['description'] == s_Heating].iat[0,0]
f_HeatingQC = HeatingQC_DF[HeatingQC_DF['description'] == s_HeatingQC].iat[0,0]
f_CentralAir = CentralAir_DF[CentralAir_DF['description'] == s_CentralAir].iat[0,0]
f_Electrical = Electrical_DF[Electrical_DF['description'] == s_Electrical].iat[0,0]
f_KitchenQual = KitchenQual_DF[KitchenQual_DF['description'] == s_KitchenQual].iat[0,0]
f_Functional = Functional_DF[Functional_DF['description'] == s_Functional].iat[0,0]
f_FireplaceQu = FireplaceQu_DF[FireplaceQu_DF['description'] == s_FireplaceQu].iat[0,0]
f_GarageType = GarageType_DF[GarageType_DF['description'] == s_GarageType].iat[0,0]
f_GarageFinish = GarageFinish_DF[GarageFinish_DF['description'] == s_GarageFinish].iat[0,0]
f_GarageQual = GarageQual_DF[GarageQual_DF['description'] == s_GarageQual].iat[0,0]
f_GarageCond = GarageCond_DF[GarageCond_DF['description'] == s_GarageCond].iat[0,0]
f_PavedDrive = PavedDrive_DF[PavedDrive_DF['description'] == s_PavedDrive].iat[0,0]
f_PoolQC = PoolQC_DF[PoolQC_DF['description'] == s_PoolQC].iat[0,0]
f_Fence = Fence_DF[Fence_DF['description'] == s_Fence].iat[0,0]
f_MiscFeature = MiscFeature_DF[MiscFeature_DF['description'] == s_MiscFeature].iat[0,0]
f_SaleType = SaleType_DF[SaleType_DF['description'] == s_SaleType].iat[0,0]
f_SaleCondition = SaleCondition_DF[SaleCondition_DF['description'] == s_SaleCondition].iat[0,0]

# Values of numerical characteristics
f_MSSubClass = s_MSSubClass
f_LotArea = s_LotArea
f_OverallQual = s_OverallQual
f_OverallCond = s_OverallCond
f_YearBuilt = s_YearBuilt
f_YearRemodAdd = s_YearRemodAdd
f_BsmtFinSF1 = s_BsmtFinSF1
f_BsmtFinSF2 = s_BsmtFinSF2
f_TotalBsmtSF = s_TotalBsmtSF
f_1stFlrSF = s_1stFlrSF
f_2ndFlrSF = s_2ndFlrSF
f_LowQualFinSF = s_LowQualFinSF
f_GrLivArea = s_GrLivArea
f_BsmtFullBath = s_BsmtFullBath
f_BsmtHalfBath = s_BsmtHalfBath
f_FullBath = s_FullBath
f_HalfBath = s_HalfBath
f_BedroomAbvGr = s_BedroomAbvGr
f_KitchenAbvGr = s_KitchenAbvGr
f_TotRmsAbvGrd = s_TotRmsAbvGrd
f_Fireplaces = s_Fireplaces
f_GarageCars = s_GarageCars
f_GarageArea = s_GarageArea
f_WoodDeckSF = s_WoodDeckSF
f_OpenPorchSF = s_OpenPorchSF
f_EnclosedPorch = s_EnclosedPorch
f_3SsnPorch = s_3SsnPorch
f_ScreenPorch = s_ScreenPorch
f_PoolArea = s_PoolArea
f_MoSold = s_MoSold
f_YrSold = s_YrSold

string_c_old_tuple_logic = """
# --- section where I use tuples

#MSSubClass_tuple = (MSSubClass_DF,f_MSSubClass,"MSSubClass")
MSZoning_tuple = (MSZoning_DF,f_MSZoning,"MSZoning")
Street_tuple = (Street_DF,f_Street,"Street")
Alley_tuple = (Alley_DF,f_Alley,"Alley")
LotShape_tuple = (LotShape_DF,f_LotShape,"LotShape")
LandContour_tuple = (LandContour_DF,f_LandContour,"LandContour")
Utilities_tuple = (Utilities_DF,f_Utilities,"Utilities")
LotConfig_tuple = (LotConfig_DF,f_LotConfig,"LotConfig")
LandSlope_tuple = (LandSlope_DF,f_LandSlope,"LandSlope")
Neighborhood_tuple = (Neighborhood_DF,f_Neighborhood,"Neighborhood")
Condition1_tuple = (Condition1_DF,f_Condition1,"Condition1")
Condition2_tuple = (Condition2_DF,f_Condition2,"Condition2")
BldgType_tuple = (BldgType_DF,f_BldgType,"BldgType")
HouseStyle_tuple = (HouseStyle_DF,f_HouseStyle,"HouseStyle")
RoofStyle_tuple = (RoofStyle_DF,f_RoofStyle,"RoofStyle")
RoofMatl_tuple = (RoofMatl_DF,f_RoofMatl,"RoofMatl")
Exterior1st_tuple = (Exterior1st_DF,f_Exterior1st,"Exterior1st")
Exterior2nd_tuple = (Exterior2nd_DF,f_Exterior2nd,"Exterior2nd")
MasVnrType_tuple = (MasVnrType_DF,f_MasVnrType,"MasVnrType")
ExterQual_tuple = (ExterQual_DF,f_ExterQual,"ExterQual")
ExterCond_tuple = (ExterCond_DF,f_ExterCond,"ExterCond")
Foundation_tuple = (Foundation_DF,f_Foundation,"Foundation")
BsmtQual_tuple = (BsmtQual_DF,f_BsmtQual,"BsmtQual")
BsmtCond_tuple = (BsmtCond_DF,f_BsmtCond,"BsmtCond")
BsmtExposure_tuple = (BsmtExposure_DF,f_BsmtExposure,"BsmtExposure")
BsmtFinType1_tuple = (BsmtFinType1_DF,f_BsmtFinType1,"BsmtFinType1")
BsmtFinType2_tuple = (BsmtFinType2_DF,f_BsmtFinType2,"BsmtFinType2")
Heating_tuple = (Heating_DF,f_Heating,"Heating")
HeatingQC_tuple = (HeatingQC_DF,f_HeatingQC,"HeatingQC")
CentralAir_tuple = (CentralAir_DF,f_CentralAir,"CentralAir")
Electrical_tuple = (Electrical_DF,f_Electrical,"Electrical")
KitchenQual_tuple = (KitchenQual_DF,f_KitchenQual,"KitchenQual")
Functional_tuple = (Functional_DF,f_Functional,"Functional")
FireplaceQu_tuple = (FireplaceQu_DF,f_FireplaceQu,"FireplaceQu")
GarageType_tuple = (GarageType_DF,f_GarageType,"GarageType")
GarageFinish_tuple = (GarageFinish_DF,f_GarageFinish,"GarageFinish")
GarageQual_tuple = (GarageQual_DF,f_GarageQual,"GarageQual")
GarageCond_tuple = (GarageCond_DF,f_GarageCond,"GarageCond")
PavedDrive_tuple = (PavedDrive_DF,f_PavedDrive,"PavedDrive")
PoolQC_tuple = (PoolQC_DF,f_PoolQC,"PoolQC")
Fence_tuple = (Fence_DF,f_Fence,"Fence")
MiscFeature_tuple = (MiscFeature_DF,f_MiscFeature,"MiscFeature")
SaleType_tuple = (SaleType_DF,f_SaleType,"SaleType")
SaleCondition_tuple = (SaleCondition_DF,f_SaleCondition,"SaleCondition")

temporary_big_array = np.ndarray(shape=(43,3))
Tuple_With_Everything = tuple(map(tuple,temporary_big_array))

Huge_List = list(Tuple_With_Everything)
Huge_List[0] = MSZoning_tuple
Huge_List[1] = Street_tuple
Huge_List[2] = Alley_tuple
Huge_List[3] = LotShape_tuple
Huge_List[4] = LandContour_tuple
Huge_List[5] = Utilities_tuple
Huge_List[6] = LotConfig_tuple
Huge_List[7] = LandSlope_tuple
Huge_List[8] = Neighborhood_tuple
Huge_List[9] = Condition1_tuple
Huge_List[10] = Condition2_tuple
Huge_List[11] = BldgType_tuple
Huge_List[12] = HouseStyle_tuple
Huge_List[13] = RoofStyle_tuple
Huge_List[14] = RoofMatl_tuple
Huge_List[15] = Exterior1st_tuple
Huge_List[16] = Exterior2nd_tuple
Huge_List[17] = MasVnrType_tuple
Huge_List[18] = ExterQual_tuple
Huge_List[19] = ExterCond_tuple
Huge_List[20] = Foundation_tuple
Huge_List[21] = BsmtQual_tuple
Huge_List[22] = BsmtCond_tuple
Huge_List[23] = BsmtExposure_tuple
Huge_List[24] = BsmtFinType1_tuple
Huge_List[25] = BsmtFinType2_tuple
Huge_List[26] = Heating_tuple
Huge_List[27] = HeatingQC_tuple
Huge_List[28] = CentralAir_tuple
Huge_List[29] = Electrical_tuple
Huge_List[30] = KitchenQual_tuple
Huge_List[31] = Functional_tuple
Huge_List[32] = FireplaceQu_tuple
Huge_List[33] = GarageType_tuple
Huge_List[34] = GarageFinish_tuple
Huge_List[35] = GarageQual_tuple
Huge_List[36] = GarageCond_tuple
Huge_List[37] = PavedDrive_tuple
Huge_List[38] = PoolQC_tuple
Huge_List[39] = Fence_tuple
Huge_List[40] = MiscFeature_tuple
Huge_List[41] = SaleType_tuple
Huge_List[42] = SaleCondition_tuple

Tuple_With_Everything = tuple(Huge_List)

# --- section replicating effects of LabelEncoding

where_the_encoded = Path(string_path_to_parent_directory) / "encoded_labels_in_order.csv"

encoded_labels_in_order_DF = pandas.read_csv(
    filepath_or_buffer=Path(where_the_encoded),
    sep=';'
)

all_values_encoded = np.ndarray(shape=(43),dtype=int)

for x in range(43):
    chosen_value = Huge_List[x][1]
    value_name = Huge_List[x][2]
    column_for_encoding = encoded_labels_in_order_DF[value_name]
    a = int(-1)
    for nm in column_for_encoding:
        a += 1
        if nm == chosen_value:
            all_values_encoded[x] = a
            break

# --- end of section for encoding labels

# --- end of section for tuples
"""

categorical_tuples = [
    (MSZoning_DF, f_MSZoning, "MSZoning"),
    (Street_DF, f_Street, "Street"),
    (Alley_DF, f_Alley, "Alley"),
    (LotShape_DF, f_LotShape, "LotShape"),
    (LandContour_DF, f_LandContour, "LandContour"),
    (Utilities_DF, f_Utilities, "Utilities"),
    (LotConfig_DF, f_LotConfig, "LotConfig"),
    (LandSlope_DF, f_LandSlope, "LandSlope"),
    (Condition1_DF, f_Condition1, "Condition1"),
    (Condition2_DF, f_Condition2, "Condition2"),
    (BldgType_DF, f_BldgType, "BldgType"),
    (HouseStyle_DF, f_HouseStyle, "HouseStyle"),
    (RoofStyle_DF, f_RoofStyle, "RoofStyle"),
    (RoofMatl_DF, f_RoofMatl, "RoofMatl"),
    (Exterior1st_DF, f_Exterior1st, "Exterior1st"),
    (Exterior2nd_DF, f_Exterior2nd, "Exterior2nd"),
    (MasVnrType_DF, f_MasVnrType, "MasVnrType"),
    (ExterQual_DF, f_ExterQual, "ExterQual"),
    (ExterCond_DF, f_ExterCond, "ExterCond"),
    (Foundation_DF, f_Foundation, "Foundation"),
    (BsmtQual_DF, f_BsmtQual, "BsmtQual"),
    (BsmtCond_DF, f_BsmtCond, "BsmtCond"),
    (BsmtFinType1_DF, f_BsmtFinType1, "BsmtFinType1"),
    (BsmtFinType2_DF, f_BsmtFinType2, "BsmtFinType2"),
    (Heating_DF, f_Heating, "Heating"),
    (HeatingQC_DF, f_HeatingQC, "HeatingQC"),
    (CentralAir_DF, f_CentralAir, "CentralAir"),
    (Electrical_DF, f_Electrical, "Electrical"),
    (KitchenQual_DF, f_KitchenQual, "KitchenQual"),
    (Functional_DF, f_Functional, "Functional"),
    (FireplaceQu_DF, f_FireplaceQu, "FireplaceQu"),
    (GarageType_DF, f_GarageType, "GarageType"),
    (GarageFinish_DF, f_GarageFinish, "GarageFinish"),
    (GarageQual_DF, f_GarageQual, "GarageQual"),
    (GarageCond_DF, f_GarageCond, "GarageCond"),
    (PavedDrive_DF, f_PavedDrive, "PavedDrive"),
    (PoolQC_DF, f_PoolQC, "PoolQC"),
    (Fence_DF, f_Fence, "Fence"),
    (MiscFeature_DF, f_MiscFeature, "MiscFeature"),
    (SaleType_DF, f_SaleType, "SaleType"),
    (SaleCondition_DF, f_SaleCondition, "SaleCondition"),
]

where_the_encoded = Path(string_path_to_parent_directory) / "encoded_labels_in_order.csv"
encoded_labels_in_order_DF = pandas.read_csv(
    filepath_or_buffer=where_the_encoded,
    sep=";",
    keep_default_na=False,
)

def encode_selected_value(chosen_value, column_name):
    """Return the LabelEncoder index for a selected categorical value."""
    for index, encoded_value in enumerate(encoded_labels_in_order_DF[column_name]):
        if str(encoded_value) == str(chosen_value):
            return index

    raise ValueError(
        f"Value {chosen_value!r} for categorical feature {column_name!r} "
        "was not found in encoded_labels_in_order.csv."
    )


all_values_encoded = np.array([
    encode_selected_value(chosen_value, column_name)
    for _, chosen_value, column_name in categorical_tuples
], dtype=np.float32)

encoded_by_name = {
    column_name: encoded_value
    for (_, _, column_name), encoded_value in zip(categorical_tuples, all_values_encoded)
}

string_c_old_feature_arrays = """
# Store numerical values in the order used by the data description.
all_values_numerical = np.array([
    f_LotFrontage,
    f_LotArea,
    f_YearBuilt,
    f_YearRemodAdd,
    f_MasVnrArea,
    f_BsmtFinSF1,
    f_BsmtFinSF2,
    f_BsmtUnfSF,
    f_TotalBsmtSF,
    f_1stFlrSF,
    f_2ndFlrSF,
    f_LowQualFinSF,
    f_GrLivArea,
    f_BsmtFullBath,
    f_BsmtHalfBath,
    f_FullBath,
    f_HalfBath,
    f_BedroomAbvGr,
    f_KitchenAbvGr,
    f_TotRmsAbvGrd,
    f_Fireplaces,
    f_GarageYrBlt,
    f_GarageCars,
    f_GarageArea,
    f_WoodDeckSF,
    f_OpenPorchSF,
    f_EnclosedPorch,
    f_3SsnPorch,
    f_ScreenPorch,
    f_PoolArea,
    f_MiscVal,
    f_MoSold,
    f_YrSold
], dtype=np.int64)

# Store all values in the order used by the data description.
all_values_together_in_order = np.array([
    all_values_encoded[0],   # MSZoning
    all_values_numerical[0], # LotFrontage
    all_values_numerical[1], # LotArea
    all_values_encoded[1],   # Street
    all_values_encoded[2],   # Alley
    all_values_encoded[3],   # LotShape
    all_values_encoded[4],   # LandContour
    all_values_encoded[5],   # Utilities
    all_values_encoded[6],   # LotConfig
    all_values_encoded[7],   # LandSlope
    all_values_encoded[8],   # Neighborhood
    all_values_encoded[9],   # Condition1
    all_values_encoded[10],  # Condition2
    all_values_encoded[11],  # BldgType
    all_values_encoded[12],  # HouseStyle
    all_values_numerical[2], # YearBuilt
    all_values_numerical[3], # YearRemodAdd
    all_values_encoded[13],  # RoofStyle
    all_values_encoded[14],  # RoofMatl
    all_values_encoded[15],  # Exterior1st
    all_values_encoded[16],  # Exterior2nd
    all_values_encoded[17],  # MasVnrType
    all_values_numerical[4], # MasVnrArea
    all_values_encoded[18],  # ExterQual
    all_values_encoded[19],  # ExterCond
    all_values_encoded[20],  # Foundation
    all_values_encoded[21],  # BsmtQual
    all_values_encoded[22],  # BsmtCond
    all_values_encoded[23],  # BsmtExposure
    all_values_encoded[24],  # BsmtFinType1
    all_values_numerical[5], # BsmtFinSF1
    all_values_encoded[25],  # BsmtFinType2
    all_values_numerical[6], # BsmtFinSF2
    all_values_numerical[7], # BsmtUnfSF
    all_values_numerical[8], # TotalBsmtSF
    all_values_encoded[26],  # Heating
    all_values_encoded[27],  # HeatingQC
    all_values_encoded[28],  # CentralAir
    all_values_encoded[29],  # Electrical
    all_values_numerical[9], # LowQualFinSF
    all_values_numerical[10], # GrLivArea
    all_values_numerical[11], # BsmtFullBath
    all_values_numerical[12], # BsmtHalfBath
    all_values_numerical[13], # FullBath
    all_values_numerical[14], # HalfBath
    all_values_numerical[15], # BedroomAbvGr
    all_values_numerical[16], # KitchenAbvGr
    all_values_encoded[30],  # KitchenQual
    all_values_numerical[17], # TotRmsAbvGrd
    all_values_encoded[31],  # Functional
    all_values_numerical[18], # Fireplaces
    all_values_encoded[32],  # FireplaceQu
    all_values_encoded[33],  # GarageType
    all_values_numerical[19], # GarageYrBlt
    all_values_encoded[34],  # GarageFinish
    all_values_numerical[20], # GarageCars
    all_values_numerical[21], # GarageArea
    all_values_encoded[35],  # GarageQual
    all_values_encoded[36],  # GarageCond
    all_values_encoded[37],  # PavedDrive
    all_values_numerical[22], # WoodDeckSF
    all_values_numerical[23], # OpenPorchSF
    all_values_numerical[24], # EnclosedPorch
    all_values_numerical[25], # 3SsnPorch
    all_values_numerical[26], # ScreenPorch
    all_values_numerical[27], # PoolArea
    all_values_encoded[38],  # PoolQC
    all_values_encoded[39],  # Fence
    all_values_encoded[40],  # MiscFeature
    all_values_numerical[28], # MiscVal
    all_values_numerical[29], # MoSold
    all_values_numerical[30], # YrSold
    all_values_encoded[41],  # SaleType
    all_values_encoded[42],  # SaleCondition
], dtype=np.float32)

"""

# These arrays mirror the 72 columns used to train the model. The removed
# columns are intentionally absent: Id, LotFrontage, Neighborhood, MasVnrArea,
# BsmtExposure, BsmtUnfSF, GarageYrBlt, MiscVal, and SalePrice.
all_values_numerical = np.array([
    f_MSSubClass, f_LotArea, f_OverallQual, f_OverallCond,
    f_YearBuilt, f_YearRemodAdd, f_BsmtFinSF1, f_BsmtFinSF2,
    f_TotalBsmtSF, f_1stFlrSF, f_2ndFlrSF, f_LowQualFinSF,
    f_GrLivArea, f_BsmtFullBath, f_BsmtHalfBath, f_FullBath,
    f_HalfBath, f_BedroomAbvGr, f_KitchenAbvGr, f_TotRmsAbvGrd,
    f_Fireplaces, f_GarageCars, f_GarageArea, f_WoodDeckSF,
    f_OpenPorchSF, f_EnclosedPorch, f_3SsnPorch, f_ScreenPorch,
    f_PoolArea, f_MoSold, f_YrSold
], dtype=np.float32)

all_values_together_in_order = np.array([
    f_MSSubClass,
    encoded_by_name["MSZoning"],
    f_LotArea,
    encoded_by_name["Street"],
    encoded_by_name["Alley"],
    encoded_by_name["LotShape"],
    encoded_by_name["LandContour"],
    encoded_by_name["Utilities"],
    encoded_by_name["LotConfig"],
    encoded_by_name["LandSlope"],
    encoded_by_name["Condition1"],
    encoded_by_name["Condition2"],
    encoded_by_name["BldgType"],
    encoded_by_name["HouseStyle"],
    f_OverallQual,
    f_OverallCond,
    f_YearBuilt,
    f_YearRemodAdd,
    encoded_by_name["RoofStyle"],
    encoded_by_name["RoofMatl"],
    encoded_by_name["Exterior1st"],
    encoded_by_name["Exterior2nd"],
    encoded_by_name["MasVnrType"],
    encoded_by_name["ExterQual"],
    encoded_by_name["ExterCond"],
    encoded_by_name["Foundation"],
    encoded_by_name["BsmtQual"],
    encoded_by_name["BsmtCond"],
    encoded_by_name["BsmtFinType1"],
    f_BsmtFinSF1,
    encoded_by_name["BsmtFinType2"],
    f_BsmtFinSF2,
    f_TotalBsmtSF,
    encoded_by_name["Heating"],
    encoded_by_name["HeatingQC"],
    encoded_by_name["CentralAir"],
    encoded_by_name["Electrical"],
    f_1stFlrSF,
    f_2ndFlrSF,
    f_LowQualFinSF,
    f_GrLivArea,
    f_BsmtFullBath,
    f_BsmtHalfBath,
    f_FullBath,
    f_HalfBath,
    f_BedroomAbvGr,
    f_KitchenAbvGr,
    encoded_by_name["KitchenQual"],
    f_TotRmsAbvGrd,
    encoded_by_name["Functional"],
    f_Fireplaces,
    encoded_by_name["FireplaceQu"],
    encoded_by_name["GarageType"],
    encoded_by_name["GarageFinish"],
    f_GarageCars,
    f_GarageArea,
    encoded_by_name["GarageQual"],
    encoded_by_name["GarageCond"],
    encoded_by_name["PavedDrive"],
    f_WoodDeckSF,
    f_OpenPorchSF,
    f_EnclosedPorch,
    f_3SsnPorch,
    f_ScreenPorch,
    f_PoolArea,
    encoded_by_name["PoolQC"],
    encoded_by_name["Fence"],
    encoded_by_name["MiscFeature"],
    f_MoSold,
    f_YrSold,
    encoded_by_name["SaleType"],
    encoded_by_name["SaleCondition"],
], dtype=np.float32)

# ---------------------------------------------------------
# Calculate button
# ---------------------------------------------------------

if st.button("Calculate"):

    # Put the values into the same order used when the model was trained.
    input_data = all_values_together_in_order
    string_c_8="""
    input_data = np.array([
        feature_1,
        f_MSZoning,
        f_Street,
        f_Alley
    ], dtype=np.float32)  """

    # Add batch dimension:
    # (4,) -> (1, 4)
    input_data = input_data.reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data, verbose=0)

    # Extract the single prediction
    #result = 10000
    result = prediction[0][0]

    # Display result
    st.subheader("Prediction")

    st.text_input(
        "Result",
        value=f"{result:.4f}",
        disabled=True
    )
