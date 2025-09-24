import Header from './components/Header/Header';
import Tabs from './components/Tabs/Tabs';
import InfoTab from './components/InfoTab/InfoTab';
import TestTab from './components/TestTab/TestTab';
import Footer from './components/Footer/Footer';
import './App.css';

function App() {
  return (
    <div className="App">
      <Header />
      <Tabs>
        <InfoTab title="Project Information" />
        <TestTab title="Model Testing" />
      </Tabs>
      <Footer />
    </div>
  );
}

export default App;