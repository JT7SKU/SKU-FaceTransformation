﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SKU_FaceTransformation.Models
{
    internal static class FaceSentimentAnalyzerConst

    {

        public const string WINML_MODEL_FILENAME = "emotion_ferplus.onnx";

        public const string WINML_MODEL_INPUTNAME = "Input3";

        public const string WINML_MODEL_OUTPUTNAME = "Plus692_Output_0";

        public const string SKILL_INPUTNAME_IMAGE = "InputImage";

        public const string SKILL_OUTPUTNAME_FACERECTANGLE = "FaceRectangle";

        public const string SKILL_OUTPUTNAME_FACESENTIMENTSCORES = "FaceSentimentScores";

        public static readonly List<float> ZeroFaceRectangleCoordinates = new List<float> { 0.0f, 0.0f, 0.0f, 0.0f };

        public static readonly List<float> ZeroFaceSentimentScores = new List<float> { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    }
}
