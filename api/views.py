from django.http import JsonResponse

def getRoutes(request):
    
    routes = [
        {'GET':'api/tickers'},
        {'GET':'api/tickers/id'},
        {'POST':'api/tickers/id/prices'},
        
        {'POST':'api/users/token'},
        {'POST':'api/users/token/refresh'},
        
    ]
    
    return JsonResponse(routes, safe=False)