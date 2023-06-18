import math
import numpy as np
import matplotlib.pyplot as plt



# set x's range
x = np.arange(-10, 10, 0.1)


y1 = 1 / (1 + math.e ** (-x))  # sigmoid
# y11=math.e**(-x)/((1+math.e**(-x))**2)
y11 = 1 / (2 + math.e ** (-x)+ math.e ** (x))  # sigmoid的导数

y2 = (math.e ** (x) - math.e ** (-x)) / (math.e ** (x) + math.e ** (-x))  # tanh
y22 = 1-y2*y2  # tanh函数的导数

y3 = np.where(x < 0, 0, x)  # relu
y33 = np.where(x < 0, 0, 1)  # ReLU函数导数

y4 = np.where(x < 0, 0.01*x, x)  # leaky relu
y44 = np.where(x < 0, 0.01, 1)  # Leaky ReLU函数导数

y5 = np.where(x < 0, 0.5*x, x)  # Prelu
y55 = np.where(x < 0, 0.5, 1)  # PReLU函数导数

y6 = np.where(x < 0, 1.6733 * (np.exp(x) - 1), x)  # Elu
y66 = np.where(x < 0, 1.6733 * np.exp(x), 1)  # Leaky ReLU函数导数

y7 = np.where(x < 0, 1.0507*1.6733 * (np.exp(x) - 1), 1.0507*x)  # SElu
y77 = np.where(x < 0, 1.0507*1.6733 * np.exp(x), 1.0507)  # SELu函数导数

y8 = np.exp(x)/np.sum(np.exp(x), axis=0)  # softmax

y9 = x/(1+np.exp(-x))  #Swish
y99 = (1+np.exp(-x)+x*np.exp(-x))/(1+np.exp(-x))**2

y1010 = (np.exp(x)*(4*np.exp(2*x)+np.exp(3*x)+4*(1+x)+np.exp(x)*(6+4*x)))/(2+2*np.exp(x)+np.exp(2*x))**2  #Mish

y11 = np.maximum(1*x, 2*x)  #Maxout

y12 = np.log(1 + np.exp(1*x + 2*x))  #Softplus
y1212 = 1/(1+np.exp(-x))

y13 = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))   #GELU

plt.xlim(-5, 5)
plt.ylim(-1, 2)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))


# Draw pic
# plt.plot(x, y1, label='Sigmoid', linestyle="-", color="red")
#plt.plot(x, y11, label='Sigmoid derivative', linestyle="-", color="violet")
#
# plt.plot(x, y2, label='Tanh', linestyle="-", color="blue")
#plt.plot(x, y22, label='Tanh derivative', linestyle="-", color="violet")
#
#plt.plot(x, y3, label='Relu', linestyle="-", color="green")
# plt.plot(x, y33, label='Relu derivative', linestyle="-", color="violet")
#
# plt.plot(x, y4, label='Leaky Relu', linestyle="-", color="orange")
# plt.plot(x, y44, label='Leaky Relu derivative', linestyle="-", color="violet")
#
# plt.plot(x, y5, label='PRelu', linestyle="-", color="yellow")
# plt.plot(x, y55, label='PRelu derivative', linestyle="-", color="violet")
#
# plt.plot(x, y6, label='elu', linestyle="-", color="olive")
# plt.plot(x, y66, label='elu derivative', linestyle="-", color="violet")
#
# plt.plot(x, y7, label='selu', linestyle="-", color="gold")
# plt.plot(x, y77, label='selu derivative', linestyle="-", color="violet")

#
# plt.plot(x, y8, label='softmax', linestyle="-", color="LightCoral")

#
# plt.plot(x, y9, label='swish', linestyle="-", color="RosyBrown")
# plt.plot(x, y99, label='swish derivative', linestyle="-", color="violet")

#
# plt.plot(x, y10, label='mish', linestyle="-", color="Maroon")
# plt.plot(x, y1010, label='swish derivative', linestyle="-", color="violet")

#
# plt.plot(x, y11, label='maxout', linestyle="-", color="tan")

#
# plt.plot(x, y12, label='softplus', linestyle="-", color="salmon")
# plt.plot(x, y1212, label='softplus derivative', linestyle="-", color="violet")

#
plt.plot(x, y13, label='GELU', linestyle="-", color="lime")

# Title
# plt.legend(['Sigmoid', 'Tanh', 'Relu'])
# plt.legend(['Sigmoid derivative'])  # y1 y11
#plt.legend(['Relu derivative'])  # y2 y22
# plt.legend(['Tanh derivative'])  # y3 y33
# plt.legend(['Leaky Relu'])  # y4 y44
# plt.legend(['PRelu derivative'])  # y5 y55
# plt.legend(['ELU derivative'])  # y6 y66
# plt.legend(['SELU derivative'])
# plt.legend(['Softmax'])
# plt.legend(['Swish derivative'])
# plt.legend(['Mish derivative'])
# plt.legend(['Maxout derivative'])
plt.legend(['GELU'])

# plt.legend(['Sigmoid', 'Sigmoid derivative', 'Relu', 'Relu derivative', 'Tanh', 'Tanh derivative'])  # y3 y33
# plt.legend(loc='upper left')  # 将图例放在左上角



plt.show()
