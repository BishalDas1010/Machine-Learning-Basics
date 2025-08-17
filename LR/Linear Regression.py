from ast import Return
import pandas as pd
import matplotlib as plt 
from sklearn.model_selection import train_test_split


df =pd.read_csv('cgpa_package_dataset.csv')
x = df[['CGPA']]
y = df[['Package(LPA)']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#model
class lr:
  def fit(self,x_train,y_train):
    avg_x = 0
    avg_y =0
    for value in x_train['CGPA']:
      avg_x = avg_x + value
    for value in y_train['Package(LPA)']:
      avg_y= avg_y + value


    print(avg_y/y.size)

    print(avg_x/x.size)


    neu = 0 
    deo = 0 

    for i in range(len(x_train['CGPA'])):
      neu += (x_train['CGPA'].iloc[i] - avg_x) * (y_train['Package(LPA)'].iloc[i] - avg_y)
      deo += (x_train['CGPA'].iloc[i] - avg_x) ** 2

    self.slope = neu / deo
    self.intercept = avg_y - slope * avg_x

    print("Slope:", self.slope)
    print("Intercept:", self.intercept)

  def pred(self,x_value)->float:
    y_pread = self.slope * x_value+self.intercept
    return  y_pread



model = lr()
model.fit(x_train=x_train,y_train= y_train)
custom_input = float(input("input the CGPA :"))

predicted =model.pred(custom_input)
print(predicted)



#test the Model 