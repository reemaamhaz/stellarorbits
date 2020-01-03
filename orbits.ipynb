#import necessary modules and libraries
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

#import necessary modules and libraries
from astropy.table import Table
#get the raw data from a file stored on my local drive and take it and put it in a table using pandas
stellar_data_raw = Table.read('../Downloads/apjaa5c41t3_mrt.txt',
                              format='cds').to_pandas().set_index('Star')
#relabel the columns so it's easier to read because the abbreviations in the data set were too short and non-descriptive
# +/- indicates the margin of error/uncertainty in the data
stellar_data = stellar_data_raw.rename(columns = {'a': 'Distance from SagA*', 'e_a':'+/- D', 'e':'Eccentricity',
'i':'Inclination', 'e_i':'+/- I', '{Omega}':'Omeg','e_{Omega}':'+/- O', 'w':'Longitude', 'e_w': '+/- Longitude', 
                                    'Tp': 'Epoch', 'e_Tp':'+/- Epoch','e_e':'+/- E','Per':'Orbital Per', 'e_Per':'+/- Per'})
#show the first ten rows of the data
stellar_data[:10]

# here I am sorting the orbital periods from shortest to longest
stellar_data_sort = stellar_data.sort_values('Orbital Per', ascending=True);
#prints the sorted list of stars
stellar_data_sort

# group the stars with distances less than 1 in one group and the ones greater than 1 in another group
close_star = stellar_data[stellar_data['Distance from SagA*'] < 1].groupby('Distance from SagA*')
# show the first group 
close_star.first()


#creates a line graph with distance (independent var) as the x-axis and oribtal period as the y-axis (dependent var)
stellar_data.set_index('Distance from SagA*')['Orbital Per'].plot.line();

#creates a scatter plot with distance (independent var) as the x-axis and oribtal period as the y-axis (dependent var)
stellar_data.plot.scatter(x='Distance from SagA*', y='Orbital Per')

#finds the correlation between the distance to Sagittarius
#OLS was giving weird data points so I'm using this value instead 
stellar_data['Distance from SagA*'].corr(stellar_data['Orbital Per'])

#gets the regression results of the independent and dependent variable
result = smf.ols('a ~ Per', data=stellar_data_raw).fit()
# this gives a summary of the results
result.summary()

#create a residual plot using the two variables from the raw data set 
sns.residplot('a','Per',data=stellar_data_raw, lowess=True, color='darkblue')
#show the plot
plt.show()
