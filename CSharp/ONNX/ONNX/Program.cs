using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ONNX
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello ONNX!");
            var r = new Random();
            int input_size = 10;
            var inArray = new List<float>();
            for (int i = 0; i < input_size; i++)
                inArray.Add( (float)r.NextDouble() );

            var filename = @"../../../../../../simple_nn.onnx";
            var session = new InferenceSession(filename);
            var inputName = session.InputMetadata.Keys.Single();
            var modelInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<float>(inArray.ToArray(), new int[]{ 1, 10 }))
            };

            var result = session.Run(modelInput);
            var output = ((DenseTensor<float>)result.Single().Value).ToArray();

            Console.WriteLine("=============================");
            Console.WriteLine("ONNX Input:");
            foreach (var item in inArray)
                Console.WriteLine(item);
            Console.WriteLine("=============================");
            Console.WriteLine("ONNX Output:");
            foreach (var item in output)
                Console.WriteLine(item);
            
        }
    }
}
