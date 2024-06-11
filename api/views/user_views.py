from typing import Any, Dict
from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from django.shortcuts import get_object_or_404


from api.serializers import UserSerializer, UserSerializerWithToken
from custom_user.models import User


from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView

from django.contrib.auth.hashers import make_password
from rest_framework import status


class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    # return additional token information
    def validate(self, attrs: Dict[str, Any]) -> Dict[str, str]:
            data = super().validate(attrs)

            serializer=UserSerializerWithToken(self.user).data
            
            for k, v in serializer.items():
                data[k] = v
           

            return data

class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


@api_view(['POST'])
def registerUser(request):
    data = request.data
    print("data>>>>>", data)
    
    try:
        if User.objects.filter(email=data['email']).exists():
            return Response({'detail': 'User with this email already exists'}, status=status.HTTP_400_BAD_REQUEST)
        
        user = User.objects.create(
            first_name=data['name'],
            email=data['email'],
            password=make_password(data['password'])
        )
        
        serializer = UserSerializerWithToken(user, many=False)
        return Response(serializer.data)
    
    except KeyError as e:
        return Response({'detail': f'Missing field: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    
@api_view(['PUT'])
@permission_classes([IsAuthenticated,])
def updateUserProfile(request):
    user = request.user
    serializer = UserSerializerWithToken(user, many=False)
    
    data = request.data
    user.first_name = data['name']
    user.email = data['email']
    
    if data['password'] != '':
        user.password = make_password(data['password'])
        
    user.save()
    
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated,])
def getUserProfile(request):
    user = request.user
    serializer = UserSerializerWithToken(user, many=False)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAdminUser])
def getUsers(request):
    users = User.objects.all()
    serializer = UserSerializerWithToken(users, many=True)
    return Response(serializer.data)