/**
 * SecureFace AI (React Native) - App Root
 * Navigation setup with dark theme.
 */

import React from 'react';
import {NavigationContainer, DefaultTheme} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import HomeScreen from './src/screens/HomeScreen';
import RegisterFaceScreen from './src/screens/RegisterFaceScreen';
import VerifyFaceScreen from './src/screens/VerifyFaceScreen';

const Stack = createNativeStackNavigator();

const DarkTheme = {
  ...DefaultTheme,
  dark: true,
  colors: {
    ...DefaultTheme.colors,
    primary: '#3b82f6',
    background: '#0f172a',
    card: '#1e293b',
    text: '#f1f5f9',
    border: '#334155',
    notification: '#22c55e',
  },
};

const screenOptions = {
  headerStyle: {backgroundColor: '#1e293b'},
  headerTintColor: '#f1f5f9',
  headerTitleStyle: {fontWeight: '700' as const, fontSize: 17},
  headerShadowVisible: false,
  contentStyle: {backgroundColor: '#0f172a'},
};

function App(): React.JSX.Element {
  return (
    <NavigationContainer theme={DarkTheme}>
      <Stack.Navigator screenOptions={screenOptions}>
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{title: 'SecureFace AI'}}
        />
        <Stack.Screen
          name="Register"
          component={RegisterFaceScreen}
          options={{title: 'Register Identity'}}
        />
        <Stack.Screen
          name="Verify"
          component={VerifyFaceScreen}
          options={{title: 'Verify Identity'}}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default App;
