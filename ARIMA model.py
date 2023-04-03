#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import warnings
import pandas as pd
import statsmodels.api as sm
import  matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.style.use('fivethirtyeight')


# In[65]:


warnings.filterwarnings("ignore")
Silver = pd.read_excel('C:/Users/07ser/OneDrive/Masaüstü/silver_data.xlsx')
df=Silver.set_index('Date')
df.index


# In[66]:


Silver= df.iloc[:,0]


# In[67]:


Silver


# In[68]:


Silver.plot()


# In[69]:


Silver_Price=df.iloc[:,0]


# In[70]:


Silver_Price


# In[71]:


diff_silver= Silver_Price.diff()


# In[72]:


diff_silver.plot()


# In[73]:


log_silver = np.log(Silver)


# In[74]:


diff_log_silver=log_silver.diff()


# In[75]:


diff_log=np.log(Silver).diff()
return_silver=diff_log.dropna()
return_silver_percent=return_silver*100
return_silver_percent.plot()


# In[76]:


plot_acf(Silver_Price)


# In[77]:


plot_pacf(Silver_Price)


# In[78]:


plot_acf(Silver_Price.diff().dropna(),lags=50) 
plot_pacf(Silver_Price.diff().dropna(),lags=20)


# In[79]:


training=Silver_Price[0:1700]
test=Silver_Price[1701:2019]


# In[80]:



mod=sm.tsa.statespace.SARIMAX(Silver_Price,trend='n',order=(1,1,0),seasonal_order=(0,0,0,0),enforce_stationarity=True, enforce_invertibility=True)
mod=sm.tsa.statespace.SARIMAX(Silver_Price,trend='n',order=(1,1,0),seasonal_order=(0,0,0,0))


# In[81]:


results=mod.fit(disp=False)
print(results.summary())
modAR1_aic=results.aic
modAR1_bic=results.bic


# In[82]:


residuals=results.resid
residuals=residuals.iloc[1:-1]
plot_acf(residuals,lags=10)


# In[83]:


# pseudo prediction 
pred=results.get_prediction(start=1701,end=1900, dynamic=True)


# In[84]:


pred_pseudo=pred.predicted_mean
mae1=abs(Silver_Price-pred_pseudo).mean() 
print(mae1)
mape1=100*(abs(Silver_Price-pred_pseudo)/Silver_Price).mean()
print(mape1)
fitted_values=results.fittedvalues


# In[85]:


mape1


# In[86]:


#model 2 ARIMA(2,1,0)


# In[87]:


mod2=sm.tsa.statespace.SARIMAX(Silver_Price,trend='n',order=(2,1,0),seasonal_order=(0,0,0,0),enforce_stationarity=True, enforce_invertibility=True)
results2=mod2.fit(disp=False)
print(results2.summary())


# In[88]:


residuals2=results2.resid
residuals2=residuals2.iloc[1:-1]
plot_acf(residuals2,lags=10)
pred2=results2.get_prediction(start=1701,end=2047,dynamic=True)
pred_pseudo2=pred2.predicted_mean


# In[89]:


mape2=100*(abs(Silver_Price - pred_pseudo2)/Silver_Price).mean()
mae2=abs(Silver_Price - pred_pseudo2).mean()


# In[90]:


print(mape2)


# In[91]:


#model 3 ARMA(1,1,1)


# In[92]:


mod3=sm.tsa.statespace.SARIMAX(Silver_Price,trend='n',order=(1,1,1),seasonal_order=(0,0,0,0),enforce_stationarity=True, enforce_invertibility=True)
results3=mod3.fit(disp=False)
print(results3.summary())


# In[93]:


residuals3=results3.resid
residuals3=residuals3.iloc[1:-1]
plot_acf(residuals3,lags=10)
pred3=results3.get_prediction(start=1701,end=2047, dynamic=True)
pred_pseudo3=pred3.predicted_mean


# In[94]:


mape3=100*(abs(Silver_Price-pred_pseudo3)/Silver_Price).mean()
mape3


# In[95]:


#model 4 ARMA(2,1,1)


# In[96]:


mod4=sm.tsa.statespace.SARIMAX(Silver_Price,trend='n',order=(2,1,1),seasonal_order=(0,0,0,0),enforce_stationarity=True, enforce_invertibility=True)
results4=mod4.fit(disp=False)
print(results4.summary())


# In[97]:


residuals4=results4.resid
residuals4=residuals4.iloc[1:-1]
plot_acf(residuals4,lags=10)                                         
pred4=results4.get_prediction(start=1701,end=2047, dynamic=True)
pred_pseudo4=pred4.predicted_mean


# In[98]:


mape4=100*(abs(Silver_Price-pred_pseudo4)/Silver_Price).mean()
mape4


# In[99]:


#model 5 ARIMA(3,1)


# In[100]:


mod5=sm.tsa.statespace.SARIMAX(Silver_Price,trend='n',order=(3,1,1),seasonal_order=(0,0,0,0),enforce_stationarity=True, enforce_invertibility=True)
results5=mod5.fit(disp=False)
print(results5.summary())


# In[101]:


residuals5=results5.resid
residuals5=residuals5.iloc[1:-1]
plot_acf(residuals5,lags=10)                                         
pred5=results5.get_prediction(start=1701,end=2047, dynamic=True)
pred_pseudo5=pred5.predicted_mean
mae5=abs(Silver_Price-pred_pseudo5).mean()
mape5=100*(abs(Silver_Price-pred_pseudo5)/Silver_Price).mean()
mape5


# In[102]:


#model 6 ARIMA(4,1,1)
mod6=sm.tsa.statespace.SARIMAX(Silver_Price,trend='n',order=(4,1,1),seasonal_order=(0,0,0,0),enforce_stationarity=True, 
                                  enforce_invertibility=True)
results6=mod6.fit(disp=False)
print(results6.summary())


# In[103]:


residuals6=results6.resid
residuals6=residuals6.iloc[1:-1]
plot_acf(residuals6,lags=10)   
pred6=results6.get_prediction(start=1701,end=2047, dynamic=True)
pred_pseudo6=pred6.predicted_mean
mae6=abs(Silver_Price-pred_pseudo5).mean()
mape6=100*(abs(Silver_Price-pred_pseudo5)/Silver_Price).mean()
mape6


# In[116]:


#real prediction / out of sample predictions 
tpred_real=results3.get_prediction(start=pd.to_datetime('2022-12-16'),end=pd.to_datetime('2022-12-23'), dynamic=True,)
pred_ci=tpred_real.conf_int() ## confidence interval
pred_real=tpred_real.predicted_mean## gives the mean of the forecasted values


# In[117]:


SARIMAX_forecast = round(results3.forecast(steps =8 ), 2)
idx = pd.date_range('2022-12-16', '2022-12-23', freq='D')
SARIMAX_forecast = pd.DataFrame(list(zip(list(idx),list(SARIMAX_forecast))), 
                                columns=['Date','ForecastValue']).set_index('Date')
SARIMAX_forecast .plot()


# In[118]:


### plotting observed and forecast values

ax = Silver.plot(label='observed', figsize=(18, 4))
pred_real.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Values')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




