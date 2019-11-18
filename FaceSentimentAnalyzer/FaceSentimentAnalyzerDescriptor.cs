using Microsoft.AI.Skills.SkillInterfacePreview;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading.Tasks;
using Windows.Foundation;
using Windows.Graphics.Imaging;

namespace SKU_FaceTransformation.Models
{
    public sealed class FaceSentimentAnalyzerDescriptor :ISkillDescriptor
    {
        
        // Member variables to hold input and output descriptions
        private List<ISkillFeatureDescriptor> m_inputSkillDesc;
        private List<ISkillFeatureDescriptor> m_outputSkillDesc;

        // Properties required by the interface
        public IReadOnlyList<ISkillFeatureDescriptor> InputFeatureDescriptors => m_inputSkillDesc;
        public IReadOnlyDictionary<string, string> Metadata => null;
        public IReadOnlyList<ISkillFeatureDescriptor> OutputFeatureDescriptors => m_outputSkillDesc;

        // Constructor
        public FaceSentimentAnalyzerDescriptor()
        {
            Information = SkillInformation.Create(
                    "FaceSentimentAnalyzer", // Name
                    "Finds a face in the image and infers its predominant sentiment from a set of 8 possible labels", // Description
                    new Guid(0xf8d275ce, 0xc244, 0x4e71, 0x8a, 0x39, 0x57, 0x33, 0x5d, 0x29, 0x13, 0x88), // Id
                    new Windows.ApplicationModel.PackageVersion() { Major = 0, Minor = 0, Build = 0, Revision = 8 }, // Version
                    "JT7-SKU Developer", // Author
                    "JT7-SKU Publishing" // Publisher
                );

            // Describe input feature
            m_inputSkillDesc = new List<ISkillFeatureDescriptor>();
            m_inputSkillDesc.Add(
                new SkillFeatureImageDescriptor(
                    "InputImage", // skill input feature name
                    "the input image onto which the sentiment analysis runs",
                    true, // isRequired (since this is an input, it is required to be bound before the evaluation occurs)
                    -1, // width
                    -1, // height
                    -1, // maxDimension
                    BitmapPixelFormat.Nv12,
                    BitmapAlphaMode.Ignore)
            );

            // Describe first output feature
            m_outputSkillDesc = new List<ISkillFeatureDescriptor>();
            m_outputSkillDesc.Add(
                new SkillFeatureTensorDescriptor(
                    "FaceRectangle", // skill output feature name
                    "a face bounding box in relative coordinates (left, top, right, bottom)",
                    false, // isRequired (since this is an output, it automatically get populated after the evaluation occurs)
                    new List<long>() { 4 }, // tensor shape
                    SkillElementKind.Float)
                );

            // Describe second output feature
            m_outputSkillDesc.Add(
                new SkillFeatureTensorDescriptor(
                    FaceSentimentAnalyzerConst.SKILL_OUTPUTNAME_FACESENTIMENTSCORES, // skill output feature name
                    "the prediction score for each class",
                    false, // isRequired (since this is an output, it automatically get populated after the evaluation occurs)
                    new List<long>() { 1, 8 }, // tensor shape
                    SkillElementKind.Float)
                );
        }
        public IAsyncOperation<ISkill> CreateSkillAsync()
        {
            return AsyncInfo.Run(async (token) =>
            {
                // Retrieve the available execution devices
                var supportedDevices = await GetSupportedExecutionDevicesAsync();
                ISkillExecutionDevice deviceToUse = supportedDevices.First();

                // Either use the first device returned (CPU) or the highest performing GPU
                int powerIndex = int.MaxValue;
                foreach (var device in supportedDevices)
                {
                    if (device.ExecutionDeviceKind == SkillExecutionDeviceKind.Gpu)
                    {
                        var directXDevice = device as SkillExecutionDeviceDirectX;
                        if (directXDevice.HighPerformanceIndex < powerIndex)
                        {
                            deviceToUse = device;
                            powerIndex = directXDevice.HighPerformanceIndex;
                        }
                    }
                }
                return await CreateSkillAsync(deviceToUse);
            });
        }

        public IAsyncOperation<IReadOnlyList<ISkillExecutionDevice>> GetSupportedExecutionDevicesAsync()
        {
            return AsyncInfo.Run(async (token) =>
            {
                return await Task.Run(() =>
                {
                    var result = new List<ISkillExecutionDevice>();

                    // Add CPU as supported device
                    result.Add(SkillExecutionDeviceCPU.Create());

                    // Retrieve a list of DirectX devices available on the system and filter them by keeping only the ones that support DX12+ feature level
                    var devices = SkillExecutionDeviceDirectX.GetAvailableDirectXExecutionDevices();
                    var compatibleDevices = devices.Where((device) => (device as SkillExecutionDeviceDirectX).MaxSupportedFeatureLevel >= D3DFeatureLevelKind.D3D_FEATURE_LEVEL_12_0);
                    result.AddRange(compatibleDevices);

                    return result as IReadOnlyList<ISkillExecutionDevice>;
                });
            });
        }

        public IAsyncOperation<ISkill> CreateSkillAsync(ISkillExecutionDevice executionDevice)
        {
            return AsyncInfo.Run(async (token) =>
            {
                // Create a skill instance with the executionDevice supplied
                var skillInstance = await FaceSentimentAnalyzerSkill.CreateAsync(this, executionDevice);

                return skillInstance as ISkill;
            });
        }

        public SkillInformation Information { get; private set; }


    }
}
