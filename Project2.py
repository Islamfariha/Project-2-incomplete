#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sqlite3
import numpy as np
import pandas as pd


# In[7]:


import os

current_working_directory = os.getcwd()
print(f"Current working directory: {current_working_directory}")


# In[8]:


import sqlite3
import pandas as pd

# Correct path to the SQLite database file, as verified from previous successful executions

# Connect to the SQLite database
conn = sqlite3.connect('/Users/farihaislam/Documents/Machine learning and dsci bc udemy/FPA_FOD_20170508.sqlite')

# Read the latitude and longitude data from the 'Fires' table
df = pd.read_sql_query("SELECT latitude, longitude FROM fires;", conn)

# Make sure to close the database connection after your operations are done
conn.close()

# Display the first few rows of the dataframe to confirm successful data retrieval
print(df.head())


# In[9]:


lats = df['LATITUDE'].values
lons = df['LONGITUDE'].values


# In[10]:


# bounding box of united states
# bbox_ll = [24.356308, -124.848974]
# bbox_ur = [49.384358, -66.885444] 

bbox_ll = [24.0, -125.0]
bbox_ur = [50.0, -66.0] 

# geographical center of united states
lat_0 = 39.833333
lon_0 = -98.583333

# compute appropriate bins to aggregate data
# nx is number of bins in x-axis, i.e. longitude
# ny is number of bins in y-axis, i.e. latitude
nx = 80
ny = 40

# form the bins
lon_bins = np.linspace(bbox_ll[1], bbox_ur[1], nx)
lat_bins = np.linspace(bbox_ll[0], bbox_ur[0], ny)

# aggregate the number of fires in each bin, we will only use the density
density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])

# get the mesh for the lat and lon
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)


# In[11]:


# # Here adding one row and column at the end of the matrix, so that 
# # density has same dimension as lat_bins_2d, lon_bins_2d, otherwise, 
# # using shading='gouraud' will raise error
density = np.hstack((density,np.zeros((density.shape[0],1))))
density = np.vstack((density,np.zeros((density.shape[1]))))


# In[12]:


pip install Cartopy


# In[13]:


pip install scipy


# In[14]:


import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter

# Recalculate bins if necessary (increase for smoother contours)
nx = 160  # Double the number of longitude bins for finer resolution
ny = 80  # Double the number of latitude bins for finer resolution

# Assuming 'lats' and 'lons' variables are your latitude and longitude data points

# Recalculate the bins and density
lon_bins = np.linspace(-125, -66, nx+1)  # +1 because these are bin edges
lat_bins = np.linspace(24, 50, ny+1)  # +1 because these are bin edges
density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])

# Create the map projection
projection = ccrs.AlbersEqualArea(central_longitude=lon_0, central_latitude=lat_0)

# Create figure and axis objects
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=projection)

# Set the extent of the map
ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

# Add land, coastlines, and state features
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Define the coordinate mesh for plotting
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins[:-1], lat_bins[:-1])

# Plot the fire density data
density_smoothed = gaussian_filter(density, sigma=2)  # Apply a Gaussian filter to smooth
fire_plot = ax.pcolormesh(lon_bins_2d, lat_bins_2d, density_smoothed, 
                          transform=ccrs.PlateCarree(), 
                          cmap='coolwarm', shading='auto')

# Add a colorbar
cbar = plt.colorbar(fire_plot, orientation='vertical', pad=0.02, aspect=50)
cbar.set_label('Fire Density')

# Define gridline options and label formatting
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlines = True
gl.ylines = True
gl.xlabel_style = {'size': 15, 'color': 'gray'}
gl.ylabel_style = {'size': 15, 'color': 'gray'}

plt.show()


# - ### The color scale on the right-hand side of the plot represents the range of fire density values, with darker and warmer colors typically indicating higher concentrations of wildfires. Areas with darker reds and oranges show regions with a higher density of fires, suggesting these regions experienced more frequent wildfires over the period from which this data was collected.The lighter or blue regions indicate fewer wildfires, and the white areas might represent no recorded fires or a density below the threshold being visualized.The state boundaries are outlined, allowing for an easy geographical reference to identify which areas within individual states are more prone to wildfires.
# 

# In[15]:


pip install bokeh


# In[16]:


import sqlite3
import pandas as pd
import numpy as np
import colorcet as cc
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LogColorMapper


# In[17]:


cnx = sqlite3.connect('/Users/farihaislam/Documents/Machine learning and dsci bc udemy/FPA_FOD_20170508.sqlite')
df = pd.read_sql_query("SELECT LATITUDE, LONGITUDE, FIRE_SIZE, STATE FROM fires", cnx)
df.head(5)


# In[18]:


pd.options.mode.chained_assignment = None


# In[19]:


new = df.loc[(df.loc[:,'STATE']!='AK') & (df.loc[:,'STATE']!='HI') & (df.loc[:,'STATE']!='PR')]


# In[20]:


new.loc[:,'LATITUDE'] = ((new.loc[:,'LATITUDE']*10).apply(np.floor))/10
new.loc[:,'LONGITUDE'] = ((new.loc[:,'LONGITUDE']*10).apply(np.floor))/10
new.loc[:,'LL_COMBO'] = new.loc[:,'LATITUDE'].map(str) + '-' + new.loc[:,'LONGITUDE'].map(str)
grouped = new.groupby(['LL_COMBO', 'LATITUDE', 'LONGITUDE'])


# In[21]:


number_of_wf = grouped['FIRE_SIZE'].agg(['count']).reset_index()
number_of_wf.head(5)


# In[22]:


size_of_wf = grouped['FIRE_SIZE'].agg(['mean']).reset_index()
size_of_wf.head(5)


# In[93]:


from bokeh.models import LogColorMapper, ColorBar
from bokeh.plotting import figure, show, output_notebook, ColumnDataSource
from bokeh.transform import transform

# Assuming number_of_wf is already defined and contains 'count', 'LATITUDE', 'LONGITUDE'

# Create a custom fire-like color palette
fire_palette = ['#000000', '#330000', '#660000', '#990000', '#CC0000', '#FF0000', '#FF3300', '#FF6600', '#FF9900', '#FFCC00', '#FFFF00']

# Define a LogColorMapper with the custom palette
color_mapper = LogColorMapper(palette=fire_palette, low=min(number_of_wf['count']), high=max(number_of_wf['count']))

# Prepare the data source for Bokeh plotting
source = ColumnDataSource(number_of_wf)

# Calculate the bounds for the plot area
lon_min, lon_max = min(number_of_wf['LONGITUDE']), max(number_of_wf['LONGITUDE'])
lat_min, lat_max = min(number_of_wf['LATITUDE']), max(number_of_wf['LATITUDE'])

# Create the figure, setting the x_range and y_range to the bounds
p1 = figure(title="Number of wildfires occurring from 1992 to 2015 (lighter color means more wildfires)",
            toolbar_location=None, plot_width=600, plot_height=400,
            x_range=(lon_min, lon_max), y_range=(lat_min, lat_max))

# Set background and grid line color
p1.background_fill_color = "black"
p1.grid.grid_line_color = None
p1.axis.visible = False

# Increase the size of the points
point_size = 3  # Adjust as needed

# Add the circles to the plot with the custom color mapper
glyph = p1.circle('LONGITUDE', 'LATITUDE', source=source,
          color=transform('count', color_mapper),
          size=point_size)

# Add a color bar to the plot
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0,0))
p1.add_layout(color_bar, 'right')

# Show the plot
output_notebook()
show(p1)


# In[92]:


from bokeh.models import LogColorMapper, ColorBar
from bokeh.plotting import figure, show, output_notebook, ColumnDataSource
from bokeh.transform import transform

# Assuming `size_of_wf` DataFrame is already created and contains 'mean', 'LATITUDE', 'LONGITUDE'

# Create the color mapper for average size
size_color_mapper = LogColorMapper(palette=fire_palette, low=min(size_of_wf['mean']), high=max(size_of_wf['mean']))

# Prepare the data source for Bokeh plotting
source_size = ColumnDataSource(size_of_wf)

# Create the figure for the average size of wildfires
p2 = figure(title="Average size of wildfires occurring from 1992 to 2015 (lighter color means bigger fire)",
            toolbar_location=None, plot_width=600, plot_height=400)

# Set background and grid line color
p2.background_fill_color = "black"
p2.grid.grid_line_color = None
p2.axis.visible = False

# Add the circles to the plot with the custom color mapper
glyph = p2.circle('LONGITUDE', 'LATITUDE', source=source_size,
          color=transform('mean', size_color_mapper),
          size=1)

# Add a color bar to the plot
color_bar = ColorBar(color_mapper=size_color_mapper, label_standoff=12, border_line_color=None, location=(0,0))
p2.add_layout(color_bar, 'right')

# Show the plot
output_notebook()
show(p2)


# - ### Time Series Analysis 

# In[81]:


from astropy.time import Time
import pandas as pd
import sqlite3

# Reconnect to the SQLite database to get the raw DISCOVERY_DATE data
conn = sqlite3.connect('/Users/farihaislam/Documents/Machine learning and dsci bc udemy/FPA_FOD_20170508.sqlite')

# Read the raw DISCOVERY_DATE data from the 'Fires' table
df = pd.read_sql_query("SELECT DISCOVERY_DATE, LATITUDE, LONGITUDE, FIRE_SIZE, STATE FROM Fires;", conn)
conn.close()

# Convert the DISCOVERY_DATE from Julian dates to standard Gregorian dates
# The Time() object from astropy handles Julian dates with fractional days
df['DISCOVERY_DATE'] = Time(df['DISCOVERY_DATE'], format='jd').to_datetime()



# In[82]:


# This will print all column names in the DataFrame
print(df.columns)


# - ### Yearly temporal Wildfire Trend and Size from 1992 to 2015

# In[83]:


import matplotlib.pyplot as plt

# Ensure the YEAR column is extracted correctly
df['YEAR'] = df['DISCOVERY_DATE'].dt.year

# Trend Over Time: Number of Wildfires
annual_wildfire_counts = df.groupby('YEAR').size()

# Trend Over Time: Average Wildfire Size
annual_wildfire_size = df.groupby('YEAR')['FIRE_SIZE'].mean()

# Plotting the number of wildfires over the years
plt.figure(figsize=(12, 6))
annual_wildfire_counts.plot(kind='line')
plt.title('Annual Wildfire Occurrences Over Time')
plt.ylabel('Number of Wildfires')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# Plotting the average size of wildfires over the years
plt.figure(figsize=(12, 6))
annual_wildfire_size.plot(kind='line', color='orange')
plt.title('Annual Average Wildfire Size Over Time')
plt.ylabel('Average Size (acres)')
plt.xlabel('Year')
plt.grid(True)
plt.show()


# - ### Zooming into the monthly trend over time

# In[87]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")

# Define the month names to use as labels
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plotting the seasonality of wildfire occurrences by month with seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_fire_count.index, y=monthly_fire_count.values, palette='autumn')
plt.xticks(np.arange(12), month_names)  # Set custom month labels
plt.title('Monthly Wildfire Occurrences')
plt.ylabel('Number of Wildfires')
plt.xlabel('Month')
plt.show()

# Plotting the seasonality of average wildfire size by month with seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_fire_size_mean.index, y=monthly_fire_size_mean.values, palette='autumn')
plt.xticks(np.arange(12), month_names)  # Set custom month labels
plt.title('Monthly Average Wildfire Size')
plt.ylabel('Average Size (acres)')
plt.xlabel('Month')
plt.show()


# In[88]:


# Overlaying a line plot on the monthly wildfire occurrences bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_fire_count.index, y=monthly_fire_count.values, color='lightcoral', alpha=0.6, label='Occurrences')
sns.lineplot(x=monthly_fire_count.index-1, y=monthly_fire_count.values, marker='o', color='red', label='Trend Line')
plt.xticks(np.arange(12), month_names)
plt.title('Monthly Wildfire Occurrences with Trend')
plt.ylabel('Number of Wildfires')
plt.xlabel('Month')
plt.legend()
plt.show()

# Overlaying a line plot on the monthly average wildfire size bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_fire_size_mean.index, y=monthly_fire_size_mean.values, color='skyblue', alpha=0.6, label='Average Size')
sns.lineplot(x=monthly_fire_size_mean.index-1, y=monthly_fire_size_mean.values, marker='o', color='blue', label='Trend Line')
plt.xticks(np.arange(12), month_names)
plt.title('Monthly Average Wildfire Size with Trend')
plt.ylabel('Average Size (acres)')
plt.xlabel('Month')
plt.legend()
plt.show()


# - ### Combining Trendlines to compare size and occurences over time

# In[89]:


# Prepare a DataFrame for plotting
monthly_data = df.groupby('MONTH').agg(count=('FIRE_SIZE', 'size'), average_size=('FIRE_SIZE', 'mean')).reset_index()
monthly_data['MONTH_NAME'] = monthly_data['MONTH'].apply(lambda x: month_names[x-1])

# Plotting
plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_data, x='MONTH_NAME', y='count', marker='o', color='blue', label='Number of Wildfires')
ax2 = plt.twinx()
sns.lineplot(data=monthly_data, x='MONTH_NAME', y='average_size', marker='o', color='orange', ax=ax2, label='Average Size (acres)')

plt.title('Wildfire Trends: Occurrences and Average Size by Month')
ax2.set_ylabel('Average Size (acres)')
plt.ylabel('Number of Wildfires')
plt.xlabel('Month')

# Handling legends for both y-axes
lines, labels = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.show()


# In[ ]:




