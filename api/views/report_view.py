from rest_framework.decorators import api_view
from rest_framework.response import Response
from ml_models.utils.access_results import read_prediction_from_Mongo

@api_view(['GET'])
def getPredictions(request, pk):
    """
    API endpoint to fetch predictions from MongoDB.

    Args:
        request: The HTTP request object.
        pk (string): The key for the MongoDB query (e.g., collection name or identifier).

    Returns:
        Response: A JSON response containing the prediction data or an error message.
    """
    try:
        # Log the pk value (useful for debugging)
        print(f"Fetching predictions for: {pk}")
        
        # Fetch predictions from MongoDB based on pk
        predictions = read_prediction_from_Mongo(pk)
        
        # Return the predictions as a JSON response
        return Response(predictions)
    except Exception as e:
        # Catch any exception and return a 500 error response
        return Response({'error': str(e)}, status=500)
