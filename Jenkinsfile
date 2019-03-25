pipeline{
	agent { docker { image 'python:3.5.1'  }  }
	environment {
		DISABLE_AUTH = 'true'
		DB_ENGINE    = 'sqlite'
 	}
	stages {
		stage('build') {
			steps {
				sh 'python --version'
				sh 'echo "Hello world"'
				sh 'zzm testfdasf3'
			}
		}
		stage('test') {
			steps {
				sh 'echo "zzm test"'
			}
		}
		stage('Deploy') {
			steps {
				echo 'Deploying'
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
