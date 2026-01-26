import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL

function App() {
  const [, setSelectedImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string>('')

  const processImage = async (file: File) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setStatus('Getting upload URL...')

    try {
      //Get presigned upload URL
      const uploadUrlResponse = await fetch(
        `${API_URL}/upload-url?filename=${encodeURIComponent(file.name)}`
      )
      
      if (!uploadUrlResponse.ok) {
        throw new Error('Failed to get upload URL')
      }
      const { upload_url, s3_key, job_id } = await uploadUrlResponse.json()
      setStatus('Uploading image to S3...')

      //Upload image directly to S3
      const uploadResponse = await fetch(upload_url, {
        method: 'PUT',
        body: file,
        headers: {
          'Content-Type': file.type,
        },
      })
      
      if (!uploadResponse.ok) {
        throw new Error('Failed to upload image')
      }

      setStatus('Submitting job to processing queue...')

      //Submit job to queue
      const submitResponse = await fetch(`${API_URL}/submit-job`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ s3_key, job_id }),
      })

      if (!submitResponse.ok) {
        throw new Error('Failed to submit job')
      }

      setStatus('Waiting for GPU worker to process...')

      //Poll for results
      const maxAttempts = 60 // 2 minutes with 2-second intervals
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise(resolve => setTimeout(resolve, 2000))

        const resultResponse = await fetch(`${API_URL}/results/${job_id}`)
        
        if (!resultResponse.ok) {
          throw new Error('Failed to get results')
        }

        const resultData = await resultResponse.json()

        if (resultData.result !== 'pending') {
          setResult(resultData.result)
          setStatus('Processing complete!')
          setLoading(false)
          return
        }

        setStatus(`Processing... (${attempt + 1}/${maxAttempts})`)
      }

      throw new Error('Processing timeout - worker may not be running')

    } catch (err: any) {
      setError(err.message || 'Something went wrong')
      setStatus('')
    } finally {
      setLoading(false)
    }
  }

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file)

      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
      }
      reader.readAsDataURL(file)

      processImage(file)
    }
  }

  const handleReset = () => {
    setSelectedImage(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setStatus('')
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full">
        <h1 className="text-2xl font-bold text-gray-800 mb-6 text-center">
          AI Image Detect
        </h1>

        {!preview ? (
          <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-500 hover:bg-gray-50 transition-colors">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <p className="mb-2 text-sm text-gray-500 font-semibold">
                Click to upload
              </p>
              <p className="text-xs text-gray-400">
                PNG, JPG, GIF up to 10MB
              </p>
            </div>
            <input
              type="file"
              className="hidden"
              accept="image/*"
              onChange={handleImageChange}
            />
          </label>
        ) : (
          <div className="space-y-4">
            <img
              src={preview}
              alt="Preview"
              className="w-full h-64 object-cover rounded-lg"
            />

            <div className="flex gap-2">
              <label className="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded cursor-pointer text-center">
                Change Image
                <input
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageChange}
                />
              </label>
              <button
                onClick={handleReset}
                className="flex-1 bg-gray-500 hover:bg-gray-600 text-white font-medium py-2 px-4 rounded"
              >
                Remove
              </button>
            </div>

            {/* Status */}
            {loading && (
              <div className="bg-blue-50 p-4 rounded">
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                  <p className="text-sm text-blue-700">{status}</p>
                </div>
              </div>
            )}

            {error && (
              <div className="bg-red-50 p-4 rounded">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}

            {result && (
              <div className="bg-green-50 p-4 rounded space-y-3">
                <p className="text-sm font-semibold text-green-800">
                  ✓ Processing Complete
                </p>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Predicted Class:</span>
                    <span className="text-lg font-bold text-green-700">
                      {(parseInt(result.predicted_class) == 0)? "AI" : "Real"}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Confidence:</span>
                    <span className="text-lg font-semibold text-green-700">
                      {(result.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                  
                  <div className="mt-3 pt-3 border-t border-green-200">
                    <p className="text-xs font-semibold text-gray-700 mb-2">
                      All Probabilities:
                    </p>
                    <div className="space-y-1">
                      {Object.entries(result.all_probabilities || {}).map(([cls, prob]) => {
                        const probability = Number(prob)
                        return (
                          <div key={cls} className="flex items-center justify-between text-xs">
                            <span className="text-gray-600">{(parseInt(cls) == 0)? "AI" : "Real"}:</span>
                            <div className="flex items-center gap-2">
                              <div className="w-24 bg-gray-200 rounded-full h-2">
                                <div 
                                  className="bg-green-500 h-2 rounded-full"
                                  style={{ width: `${(probability * 100)}%` }}
                                ></div>
                              </div>
                              <span className="text-gray-700 w-12 text-right">
                                {(probability * 100).toFixed(2)}%
                              </span>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App