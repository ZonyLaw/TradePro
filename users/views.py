from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from custom_user.models import User
from users.models import Profile

def profiles(request):
    
    profiles = Profile.objects.all()
    
    context = {"profiles": profiles}
    
    return render(request, 'users/profile.html', context)


def userProfile(request, pk):
    
    profile = Profile.objects.get(id=pk)
    
    context = {'profile': profile}
    return render(request, 'users/user-profile.html', context)


def loginUser(request):
    page = 'login'
    
    if request.user.is_authenticated:
        return redirect('profiles')
    
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        
        try:
            user = User.objects.get(email=email)
        except:
            print('Email does not exist')
            
        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('profiles')
        else:
            print('Email or passwrod is incorrect')
    
    return render(request, 'users/login_register.html')

def logoutUser(request):
    logout(request)
    return redirect('login')

def registerUser(request):
    page = 'register'
    form = UserCreationForm()
    context = {'page': page, 'form':form}
    return render(request, 'users/login_register.html', context)
    