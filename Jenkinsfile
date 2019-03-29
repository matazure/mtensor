pipeline{
	agent none
	stages {
		stage('ci'){
			parallel {
				stage('linux') {
					stages {
						stage('x86_64') {
							agent { 
								docker { 
									image 'matazure/ci4tensor:gcc-ubuntu18.04'  }
					 		}
							environment {
								CXX = 'g++'
								CC = 'gcc'
							}
							steps {
								sh './script/build.sh'
							}
						}
					}
				}
				stage('windows') {
					agent {
						label 'win10-x64'
					}
					steps {
						bat 'call ./script/build_win.bat'
					}
				}
				stage('android') {
					agent {
						docker {
							image 'matazure/ci4tensor:android-ndk-r14b'
						}
					}
					stages {
						stage('armv7') {
							stages {
								stage('build') {
									steps {
										sh './script/build_android.sh'
									}
								}
								stage('test') {
									steps {
										sh 'echo armv7-test'
									}
								}
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
