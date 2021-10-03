
using Microsoft.AI.Skills.SkillInterface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Media.FaceAnalysis;
using Windows.Storage;

namespace FacesentimentAnalyzer.Models
{
    public sealed class FaceSentimentAnalyzerSkill :ISkill
    {
        private FaceDetector m_faceDetector = null;
        private LearningModelSession m_winmlSession = null;
        public ISkillDescriptor SkillDescriptor { get; private set; }
        public ISkillExecutionDevice Device { get; private set; }
        // Constructor
        private FaceSentimentAnalyzerSkill(
                ISkillDescriptor description,
                ISkillExecutionDevice device)
        {
            SkillDescriptor = description;
            Device = device;
        }

        // ISkill Factory method
        internal static IAsyncOperation<FaceSentimentAnalyzerSkill> CreateAsync(
            ISkillDescriptor descriptor,
            ISkillExecutionDevice device)
        {
            return AsyncInfo.Run(async (token) =>
            {
                // Create instance
                var skillInstance = new FaceSentimentAnalyzerSkill(descriptor, device);

                // Instantiate the FaceDetector
                skillInstance.m_faceDetector = await FaceDetector.CreateAsync();

                // Load ONNX model and instantiate LearningModel
                var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///FaceSentimentAnalyzer/emotion_ferplus.onnx"));
                var winmlModel = LearningModel.LoadFromFilePath(modelFile.Path);

                // Create LearningModelSession
                skillInstance.m_winmlSession = new LearningModelSession(winmlModel, GetWinMLDevice(device));

                return skillInstance;
            });
        }
        // ISkillBinding Factory method
        public IAsyncOperation<ISkillBinding> CreateSkillBindingAsync()
        {
            return AsyncInfo.Run((token) =>
            {
                var completedTask = new TaskCompletionSource<ISkillBinding>();
                completedTask.SetResult(new FaceSentimentAnalyzerBinding(SkillDescriptor, Device, m_winmlSession));
                return completedTask.Task;
            });
        }
        // Skill core logic
        public IAsyncAction EvaluateAsync(ISkillBinding binding)
        {
            FaceSentimentAnalyzerBinding bindingObj = binding as FaceSentimentAnalyzerBinding;
            if (bindingObj == null)
            {
                throw new ArgumentException("Invalid ISkillBinding parameter: This skill handles evaluation of FaceSentimentAnalyzerBinding instances only");
            }

            return AsyncInfo.Run(async (token) =>
            {
                // Retrieve input frame from the binding object
                VideoFrame inputFrame = (binding[FaceSentimentAnalyzerConst.SKILL_INPUTNAME_IMAGE].FeatureValue as SkillFeatureImageValue).VideoFrame;
                SoftwareBitmap softwareBitmapInput = inputFrame.SoftwareBitmap;

                // Retrieve a SoftwareBitmap to run face detection
                if (softwareBitmapInput == null)
                {
                    if (inputFrame.Direct3DSurface == null)
                    {
                        throw (new ArgumentNullException("An invalid input frame has been bound"));
                    }
                    softwareBitmapInput = await SoftwareBitmap.CreateCopyFromSurfaceAsync(inputFrame.Direct3DSurface);
                }

                // Retrieve face rectangle output feature from the binding object
                var faceRectangleFeature = binding[FaceSentimentAnalyzerConst.SKILL_OUTPUTNAME_FACERECTANGLE];

                // Retrieve face sentiment scores output feature from the binding object
                var faceSentimentScores = binding[FaceSentimentAnalyzerConst.SKILL_OUTPUTNAME_FACESENTIMENTSCORES];
                // Run face detection and retrieve face detection result
                var faceDetectionResult = await m_faceDetector.DetectFacesAsync(softwareBitmapInput);
                // If a face is found, update face rectangle feature
                if (faceDetectionResult.Count > 0)
                {
                    // Retrieve the face bound and enlarge it by a factor of 1.5x while also ensuring clamping to frame dimensions
                    BitmapBounds faceBound = faceDetectionResult[0].FaceBox;
                    var additionalOffset = faceBound.Width / 2;
                    faceBound.X = Math.Max(0, faceBound.X - additionalOffset);
                    faceBound.Y = Math.Max(0, faceBound.Y - additionalOffset);
                    faceBound.Width = (uint)Math.Min(faceBound.Width + 2 * additionalOffset, softwareBitmapInput.PixelWidth - faceBound.X);
                    faceBound.Height = (uint)Math.Min(faceBound.Height + 2 * additionalOffset, softwareBitmapInput.PixelHeight - faceBound.Y);

                    // Set the face rectangle SkillFeatureValue in the skill binding object
                    // note that values are in normalized coordinates between [0, 1] for ease of use
                    await faceRectangleFeature.SetFeatureValueAsync(
                        new List<float>()
                        {
                        (float)faceBound.X / softwareBitmapInput.PixelWidth, // left
                        (float)faceBound.Y / softwareBitmapInput.PixelHeight, // top
                        (float)(faceBound.X + faceBound.Width) / softwareBitmapInput.PixelWidth, // right
                        (float)(faceBound.Y + faceBound.Height) / softwareBitmapInput.PixelHeight // bottom
                        });

                    // Bind the WinML input frame with the adequate face bounds specified as metadata
                    bindingObj.m_winmlBinding.Bind(
                        "Input3", // WinML input feature name defined in ONNX protobuf
                        inputFrame, // VideoFrame
                        new PropertySet() // VideoFrame bounds
                        {
                    { "BitmapBounds",
                        PropertyValue.CreateUInt32Array(new uint[]{ faceBound.X, faceBound.Y, faceBound.Width, faceBound.Height })
                    }
                        });

                    // Run WinML evaluation
                    var winMLEvaluationResult = await m_winmlSession.EvaluateAsync(bindingObj.m_winmlBinding, "");

                    // Retrieve result using the WinML output feature name defined in ONNX protobuf
                    var winMLModelResult = (winMLEvaluationResult.Outputs["Plus692_Output_0"] as TensorFloat).GetAsVectorView();

                    // Set the SkillFeatureValue in the skill binding object related to the face sentiment scores for each possible SentimentType
                    // note that we SoftMax the output of WinML to give a score normalized between [0, 1] for ease of use
                    var predictionScores = SoftMax(winMLModelResult);
                    await faceSentimentScores.SetFeatureValueAsync(predictionScores);
                }
                else // if no face found, reset output SkillFeatureValues with 0s
                {
                    await faceRectangleFeature.SetFeatureValueAsync(FaceSentimentAnalyzerConst.ZeroFaceRectangleCoordinates);
                    await faceSentimentScores.SetFeatureValueAsync(FaceSentimentAnalyzerConst.ZeroFaceSentimentScores);
                }
            });
        }
        private List<float> SoftMax(IReadOnlyList<float> inputs)
        {
            List<float> inputsExp = new List<float>();
            float inputsExpSum = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                inputsExp.Add((float)Math.Exp(input));
                inputsExpSum += inputsExp[i];
            }
            inputsExpSum = inputsExpSum == 0 ? 1 : inputsExpSum;
            for (int i = 0; i < inputs.Count; i++)
            {
                inputsExp[i] /= inputsExpSum;
            }
            return inputsExp;
        }

        /// <summary>
        /// If possible, retrieves a WinML LearningModelDevice that corresponds to an ISkillExecutionDevice
        /// </summary>
        /// <param name="executionDevice"></param>
        /// <returns></returns>
        private static LearningModelDevice GetWinMLDevice(ISkillExecutionDevice executionDevice)
        {
            switch (executionDevice.ExecutionDeviceKind)
            {
                case SkillExecutionDeviceKind.Cpu:
                    return new LearningModelDevice(LearningModelDeviceKind.Cpu);

                case SkillExecutionDeviceKind.Gpu:
                    {
                        var gpuDevice = executionDevice as SkillExecutionDeviceDirectX;
                        return LearningModelDevice.CreateFromDirect3D11Device(gpuDevice.Direct3D11Device);
                    }

                default:
                    throw new ArgumentException("Passing unsupported SkillExecutionDeviceKind");
            }
        }
    }
}
