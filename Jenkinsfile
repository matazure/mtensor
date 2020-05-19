pipeline{
    agent none
    
    options {
        timeout(time: 1, unit: 'HOURS') 
    }
    triggers {
        cron('H H(0-7) * * *')
    }
    stages {
        stage('TENSOR CI'){
            parallel {
                
                stage('linux-x64') {
                    agent { 
                        dockerfile {
                            filename 'tensor-dev-ubuntu18.04.dockerfile'
                            dir 'dockerfile'
                        }
                    }
                    environment {
                        CXX = 'g++'
                        CC = 'gcc'
                    }
                    stages {
                        stage('build') {
                            steps {
                                sh './script/build_native.sh'
                            }
                        }
                        stage('test') {
                            steps {
                                sh './build/bin/ut_host_mtensor'
                            }
                        }
                        stage('sample') {
                            steps {
                                sh './build/bin/sample_for_index'
                                sh './build/bin/sample_basic_structure'
                                sh './build/bin/sample_gradient data/lena.jpg'
                                sh './build/bin/sample_mandelbrot'
                                sh './build/bin/sample_make_lambda'
                            }
                        }
                        stage('benchmark') {
                            steps {
                                sh './build/bin/bm_host_mtensor --benchmark_min_time=1'
                            }
                        }
                        stage('archive') {
                            steps {
                                archiveArtifacts artifacts: '*.png', fingerprint: true 
                            }
                        }
                    }
                }


                stage('linux-x64-cuda') {
                    agent {
                        dockerfile {
                            filename 'tensor-dev-ubuntu18.04.dockerfile'
                            dir 'dockerfile'
                            args '--gpus all'
                        }
                    }
                    environment {
                        CXX = 'g++'
                        CC = 'gcc'
                    }
                    stages {
                        stage('build') {
                            steps {
                                sh './script/build_native.sh -DWITH_CUDA=ON'
                            }
                        }
                        stage('test') {
                            steps {
                                sh './build/bin/ut_cuda_mtensor'
                            }
                        }
                        stage('sample') {
                            steps {
                                sh './build/bin/sample_cuda_for_index'
                                sh './build/bin/sample_cuda_convolution data/lena.jpg'
                                sh './build/bin/sample_cuda_mandelbrot'
                                sh './build/bin/sample_cuda_matrix_mul'
                            }
                        }
                        stage('benchmark') {
                            steps {
                                sh './build/bin/bm_cuda_mtensor'
                            }
                        }
                        stage('archive') {
                            steps {
                                archiveArtifacts artifacts: '*.png', fingerprint: true 
                            }
                        }
                    }
                }

                stage('linux-aarch64') {
                    agent {
                        dockerfile {
                            filename 'tensor-dev-ubuntu18.04.dockerfile'
                            dir 'dockerfile'
                            args '-v /root/.ssh:/root/.ssh'
                        }
                    }
                    environment {
                        GCC_LINARO_TOOLCHAIN = '/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu'
                    }
                    stages {    
                        stage('build') {
                            steps {
                                sh './script/build-linux-aarch64.sh'
                            }
                        }
                        stage('clean-remote') {
                            when {
                                triggeredBy "TimerTrigger"
                            }
                            steps {
                                sh "ssh rk3399 rm -rf /home/pi/tensor_ci"
                            }
                        }
                        stage('test') {
                            steps {
                                sh "ssh rk3399 mkdir -p \\/home/pi/tensor_ci/${env.GIT_COMMIT}"
                                sh "scp -r ./build-linux-aarch64 rk3399:/home/pi/tensor_ci/${env.GIT_COMMIT}/"
                                sh "ssh rk3399 'cd /home/pi/tensor_ci/${env.GIT_COMMIT}/build-linux-aarch64 && ./bin/ut_host_mtensor'"
                            }
                        }
                        stage('benchmark') {
                            steps {
                                sh "ssh rk3399 'cd /home/pi/tensor_ci/${env.GIT_COMMIT}/build-linux-aarch64 && ./bin/bm_host_mtensor --benchmark_min_time=1'"
                            }
                        }
                    }
                }

                stage('linux-armv7') {
                    agent {
                        dockerfile {
                            filename 'tensor-dev-ubuntu18.04.dockerfile'
                            dir 'dockerfile'
                            args '-v /root/.ssh:/root/.ssh'
                        }
                    }

                    stages {    
                        stage('build') {
                            steps {
                                sh './script/build-linux-armv7.sh'
                            }
                        }
                        stage('clean-remote') {
                            when {
                                triggeredBy "TimerTrigger"
                            }
                            steps {
                                sh "ssh rpi4 rm -rf /home/pi/tensor_ci"
                            }
                        }
                        stage('test') {
                            steps {
                                sh "ssh rpi4 mkdir -p \\/home/pi/tensor_ci/${env.GIT_COMMIT}"
                                sh "scp -r ./build-linux-armv7 rpi4:/home/pi/tensor_ci/${env.GIT_COMMIT}/"
                                sh "ssh rpi4 'cd /home/pi/tensor_ci/${env.GIT_COMMIT}/build-linux-armv7 && ./bin/ut_host_mtensor'"
                            }
                        }
                        stage('benchmark') {
                            steps {
                                sh "ssh rpi4 'cd /home/pi/tensor_ci/${env.GIT_COMMIT}/build-linux-armv7 && ./bin/bm_host_mtensor --benchmark_min_time=1'"
                            }
                        }
                    }
                }
                                 
                stage('windows-x64') {
                    agent {
                        label 'Windows'
                    }
                    stages {
                        stage('build'){
                            steps {
                                // build with cuda, but not run cuda executable
                                // windows vmware has no nvdia card 
                                bat 'call ./script/build_windows.bat -DWITH_CUDA=ON'
                            }
                        }
                        stage('test') {
                            steps {
                                powershell './build_win/bin/Release/ut_host_mtensor.exe'
                            }
                        }
                        stage('benchmark') {
                            when {
                                triggeredBy "TimerTrigger"
                            }
                            steps {
                                powershell './build_win/bin/Release/bm_host_mtensor.exe --benchmark_min_time=1'
                            }
                        }
                    }
                }
                
                stage('android-armv7') {
                    agent {
                        dockerfile {
                            filename 'tensor-dev-ubuntu18.04.dockerfile'
                            dir 'dockerfile'
                        }
                    }
                    stages {    
                        stage('build') {
                            steps {
                                sh './script/build_android.sh'
                            }
                        }
                    }
                }
            }
        }
    }
    post {
        always {
            echo 'This will always run'
        }
        success {
            echo 'This will run only ifsuccessful'
        }
        failure {
            echo 'This will run only iffailed'
        }
        unstable {
            echo 'This will run only if th run was marked as unstable'
        }
        changed {
            echo 'This will run only if th state of the Pipeline has changed'
            echo 'For example, if thePipeline was previously failing bu is now successful'
        }
    }
}
