pipeline{
    agent none
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
                                sh './build/bin/ut_mtensor_host'
                            }
                        }
                        stage('benchmark') {
                            steps {
                                sh './build/bin/bm_mtensor_host --benchmark_min_time=1'
                            }
                        }
                    }
                }


                stage('linux-x64-cuda') {
                    agent {
                        dockerfile {
                            filename 'tensor-dev-ubuntu18.04.dockerfile'
                            dir 'dockerfile'
                            args '--runtime=nvidia'
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
                                sh './build/bin/ut_cuda'
                            }
                        }
                        stage('benchmark') {
                            steps {
                                sh './build/bin/bm_mtensor_cuda'
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
                        stage('test') {
                            steps {
                                sh "ssh rk3399 mkdir -p \\~/tensor_ci/${env.GIT_COMMIT}"
                                sh "scp -r ./build-linux-aarch64 rk3399:~/tensor_ci/${env.GIT_COMMIT}/"
                                sh "ssh rk3399 'cd ~/tensor_ci/${env.GIT_COMMIT}/build-linux-aarch64 && ./bin/ut_mtensor_host'"
                            }
                        }
                        stage('benchmark') {
                            steps {
                                sh "ssh rk3399 'cd ~/tensor_ci/${env.GIT_COMMIT}/build-linux-aarch64 && ./bin/bm_mtensor_host --benchmark_min_time=1'"
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
                        stage('test') {
                            steps {
                                sh "ssh rpi4 mkdir -p \\~/tensor_ci/${env.GIT_COMMIT}"
                                sh "scp -r ./build-linux-armv7 rpi4:~/tensor_ci/${env.GIT_COMMIT}/"
                                sh "ssh rpi4 'cd ~/tensor_ci/${env.GIT_COMMIT}/build-linux-armv7 && ./bin/ut_mtensor_host'"
                            }
                        }
                        stage('benchmark') {
                            steps {
                                sh "ssh rpi4 'cd ~/tensor_ci/${env.GIT_COMMIT}/build-linux-armv7 && ./bin/bm_mtensor_host --benchmark_min_time=1'"
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
                                powershell './build_win/bin/Release/ut_mtensor_host.exe'
                            }
                        }
                        stage('benchmark') {
                            steps {
                                powershell './build_win/bin/Release/bm_mtensor_host.exe --benchmark_min_time=1'
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
