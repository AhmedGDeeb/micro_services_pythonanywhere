from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import IrisSerializer
import joblib
import numpy as np
import sklearn

@api_view(['POST'])
def classify_iris(request):
    serializer = IrisSerializer(data=request.data)
    if serializer.is_valid():
        sepal_length = serializer.validated_data['sepal_length']
        sepal_width = serializer.validated_data['sepal_width']
        petal_length = serializer.validated_data['petal_length']
        petal_width = serializer.validated_data['petal_width']
        
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        model_with_metadata = joblib.load('static/admin/ml_models/best_knn_model_v01.pkl')

        model = model_with_metadata['model']
        targets = model_with_metadata['target_names']
        prediction = targets[model.predict(features)]
        prediction_proba = model.predict_proba(features)
        
        return Response({
            'prediction': prediction[0],
            'prediction_proba': prediction_proba[0].tolist()
        })
    return Response(serializer.errors, status=400)
