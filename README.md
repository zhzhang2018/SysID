# SysID
This is the repository for the system identification work I've been doing during summer, 2020. Note that the code was written while I was gradually familiarizing myself with the area, so the code structure is far from perfect. 

# What's the big idea?
The goal is to identify the inner workings of a system via partial observations of some variables inside it. For example, when you're pushing an object, your action can be described by using a 2-state system consisting of position (<img src="https://render.githubusercontent.com/render/math?math=x">) and velocity (<img src="https://render.githubusercontent.com/render/math?math=v">). The state dynamics would be:
<img src="https://render.githubusercontent.com/render/math?math=\dot{x} = v">,   
<img src="https://render.githubusercontent.com/render/math?math=\dot{v} = \frac{F}{m}"> , where F (force) divided by m (mass) is, of course, the acceleration of the object.  
In the above example, if you know one of the state variables above, you can easily find the other. If you know <img src="https://render.githubusercontent.com/render/math?math=x">, simply differentiate it to obtain <img src="https://render.githubusercontent.com/render/math?math=v">; if you know <img src="https://render.githubusercontent.com/render/math?math=v">, simply integrate it to get <img src="https://render.githubusercontent.com/render/math?math=x">.

However, the real life is rarely so simple and satisfying. 
If you have a continuous system with 3 or more variables, your system could exhibit chaotic behavior ([Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system)) - or if your system is discrete, then one variable itself can already give you chaos. 
In addition, many real-world dynamics are complex, high-dimensional, and generally don't have a closed-form expression. For example, can you find a way to definitely describe the stock market, the weather, the marine currents, or your physical states with a set of equations? 

This is where machine learning and neural networks come into play. Of course, we probably still can't use them to describe highly complex systems like stock markets yet, but their great power (universal approximation) can help describe the system in their own ways.
It is our hope that the network can find a way to express the system through several observation variables. It is a known fact that a delay-embedded observation offers more information - knowing x(t) and x(t-T) gives you more information than knowing x(t) alone. Right now, we use this method to reconstruct the system dynamics. Ideally, we would be able to accomplish prediction, simulation, and control altogether via this method.
# What do the files do?
The Python files serve as libraries of methods, while the Jupyter Notebook files offer interactive usage. For example, "system_dynamics.py" includes modules and methods for specifying a new dynamics that comprises of ODEs (currently done by manually creating another class), and creating state trajectory data from a provided input function. On the other hand, files like "Lorenz_delay.ipynb" uses the system_dynamics module (and others) to try to perform system identification from one observed variable in the Lorenz system. 

# How do I use them?
Start from imitating one of the Jupyter notebook files, such as the most recent Coupled_Rossler_delay.ipynb. If you want to try a different dynamics, create your own class inside embed_dynamics.py by writing the dynamics equations and specifying observable dimensions. 

Note: Because my understanding of the subject has been evolving as I wrote the code, I have been constantly expanding old functions and refactoring data structures and code organizations as well. As a result, some of the files might encounter errors, because the new code might not be backwards compatible with the old ones using them (too much work to handle). For best results, stick with the most recent files. 
