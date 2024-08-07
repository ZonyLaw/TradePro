from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from .form import CustomUserCreationForm, ProfileForm
from custom_user.models import User
from users.models import Profile


def home(request):
    context = {}
    return render(request, 'home.html', context)


def users_list(request):
    
    profiles = Profile.objects.all()
    
    context = {"profiles": profiles}
    
    return render(request, 'users/users_list.html', context)


def userAccount(request):
    profile = request.user.profile

    context = {'profile': profile}
    return render(request, 'users/account.html', context)


def userProfile(request, pk):
    
    profile = Profile.objects.get(id=pk)
    
    context = {'profile': profile}
    return render(request, 'users/user_profile.html', context)

#TODO: after login it returns the whole user list...this is wrong.
def loginUser(request):
    page = 'login'
    
    if request.user.is_authenticated:
        return redirect('users-list')
    
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        
        try:
            user = User.objects.get(email=email)
        except:
            messages.error(request, "Email does not exist")
            
        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('account')
        else:
            messages.error(request, "Email or passwrod is incorrect")
    
    return render(request, 'users/login_register.html')

def logoutUser(request):
    logout(request)
    messages.info(request, "User was successfully logged out.")
    return redirect('login')

def registerUser(request):
    page = 'register'
    form = CustomUserCreationForm()
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email = user.email.lower()
            user.save()
            
            messages.success(request, "User successfully sign up.")
            login(request, user)
            
    context = {'page': page, 'form':form}
    return render(request, 'users/login_register.html', context)
    
    
def editProfile(request):
    profile = request.user.profile
    form = ProfileForm(instance=profile)

    if request.method == 'POST':
        form = ProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            try:
                form.save()
                return redirect('user-profile', pk = profile.id)
            except:
                messages.error(request, "There is an error!")
                # return redirect('edit-account')

    context = {'form': form}
    return render(request, 'users/profile_form.html', context)


def backend_control(request):
  
    context = {}
    return render(request, 'users/backend_control.html', context)